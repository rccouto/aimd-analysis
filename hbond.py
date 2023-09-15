##############################################################################
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2013 Stanford University and the Authors
#
# Authors: Robert McGibbon
# Contributors:
#
# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.
##############################################################################


##############################################################################
# Imports
##############################################################################

from __future__ import print_function, division
import numpy as np
import mdtraj as md
from mdtraj.utils import ensure_type
from mdtraj.geometry import compute_distances, compute_angles
from mdtraj.geometry import _geometry


__all__ = ['wernet_nilsson', 'baker_hubbard', 'kabsch_sander']

##############################################################################
# Functions
##############################################################################

def wernet_nilsson(traj, target, exclude_water=True, periodic=True, sidechain_only=False):
    """Identify hydrogen bonds based on cutoffs for the Donor-H...Acceptor
    distance and angle according to the criterion outlined in [1].
    As opposed to Baker-Hubbard, this is a "cone" criterion where the
    distance cutoff depends on the angle.

    The criterion employed is :math:`r_\\text{DA} < 3.3 A - 0.00044*\\delta_{HDA}*\\delta_{HDA}`,
    where :math:`r_\\text{DA}` is the distance between donor and acceptor heavy atoms,
    and :math:`\\delta_{HDA}` is the angle made by the hydrogen atom, donor, and acceptor atoms,
    measured in degrees (zero in the case of a perfectly straight bond: D-H ... A).

    When donor the donor is 'O' and the acceptor is 'O', this corresponds to
    the definition established in [1]_. The donors considered by this method
    are NH and OH, and the acceptors considered are O and N. In the paper the only
    donor considered is OH.

    Parameters
    ----------
    traj : md.Trajectory
        An mdtraj trajectory. It must contain topology information.
    exclude_water : bool, default=True
        Exclude solvent molecules from consideration.
    periodic : bool, default=True
        Set to True to calculate displacements and angles across periodic box boundaries.
    sidechain_only : bool, default=False
        Set to True to only consider sidechain-sidechain interactions.

    Returns
    -------
    hbonds : list, len=n_frames
        A list containing the atom indices involved in each of the identified
        hydrogen bonds at each frame. Each element in the list is an array
        where each row contains three integer indices, `(d_i, h_i, a_i)`,
        such that `d_i` is the index of the donor atom, `h_i` the index
        of the hydrogen atom, and `a_i` the index of the acceptor atom involved
        in a hydrogen bond which occurs in that frame.

    Notes
    -----
    Each hydrogen bond is distinguished for the purpose of this function by the
    indices of the donor, hydrogen, and acceptor atoms. This means that, for
    example, when an ARG sidechain makes a hydrogen bond with its NH2 group,
    you might see what appear like double counting of the h-bonds, since the
    hydrogen bond formed via the H_1 and H_2 are counted separately, despite
    their "chemical indistinguishably"

    Examples
    --------
    >>> md.wernet_nilsson(t)
    array([[  0,  10,   8],
           [  0,  11,   7],
           [ 69,  73,  54],
           [ 76,  82,  65],
           [119, 131,  89],
           [140, 148, 265],
           [166, 177, 122],
           [181, 188, 231]])
    >>> label = lambda hbond : '%s -- %s' % (t.topology.atom(hbond[0]), t.topology.atom(hbond[2]))
    >>> for hbond in hbonds:
    >>>     print label(hbond)
    GLU1-N -- GLU1-OE2
    GLU1-N -- GLU1-OE1
    GLY6-N -- SER4-O
    CYS7-N -- GLY5-O
    TYR11-N -- VAL8-O
    MET12-N -- LYS20-O

    See Also
    --------
    baker_hubbard, kabsch_sander

    References
    ----------
    .. [1] Wernet, Ph., L.G.M. Pettersson, and A. Nilsson, et al.
       "The Structure of the First Coordination Shell in Liquid Water." (2004)
       Science 304, 995-999.   d
    """

    distance_cutoff = 0.33
    angle_const = 0.000044
    angle_cutoff = 45

    if traj.topology is None:
        raise ValueError('wernet_nilsson requires that traj contain topology '
                         'information')

    
    # Get the target surroundings
    SELEC=f'resname {target[0:3]}'
    chrome = traj.topology.select(SELEC)
    sur_resname = compute_neighbours(traj,chrome, 0.5, True)

    # Get the possible donor-hydrogen...acceptor triplets
    bond_triplets = _get_bond_triplets(traj.topology, sur_resname,
        exclude_water=exclude_water, sidechain_only=sidechain_only)

    # Compute geometry
    mask, distances, angles = _compute_bounded_geometry(traj, bond_triplets,
        distance_cutoff, [0, 2], [2, 0, 1], periodic=periodic)

    # Update triplets under consideration
    bond_triplets = bond_triplets.compress(mask, axis=0)

    # Calculate the true cutoffs for distances
    cutoffs = distance_cutoff - angle_const * (angles * 180.0 / np.pi) ** 2

    # Find triplets that meet the criteria
    presence = np.logical_and(distances < cutoffs, angles < angle_cutoff)

    return [bond_triplets.compress(present, axis=0) for present in presence]


def _get_bond_triplets(topology, sur_resname, exclude_water=True, sidechain_only=False):

    def can_participate(atom, sur_resname):

        # Filter waters
        if exclude_water and atom.residue.is_water:
            return False
        # Filter non-sidechain atoms
        if sidechain_only and not atom.is_sidechain:
            return False
        # Filter residues that are not surrounding the target 
        if sur_resname and str(atom.residue) not in sur_resname:
            return False
        # Otherwise, accept it
        return True

    def get_donors(e0, e1):
        # Find all matching bonds
        elems = set((e0, e1))
        atoms = [(one, two) for one, two in topology.bonds
            if set((one.element.symbol, two.element.symbol)) == elems]

        # Filter non-participating atoms
        atoms = [atom for atom in atoms
            if can_participate(atom[0], sur_resname) and can_participate(atom[1], sur_resname)]

        # Get indices for the remaining atoms
        indices = []
        for a0, a1 in atoms:
            pair = (a0.index, a1.index)
            # make sure to get the pair in the right order, so that the index
            # for e0 comes before e1
            if a0.element.symbol == e1:
                pair = pair[::-1]
            indices.append(pair)

        return indices

    # Check that there are bonds in topology
    nbonds = 0
    for _bond in topology.bonds:
        nbonds += 1
        break # Only need to find one hit for this check (not robust)
    if nbonds == 0:
        raise ValueError('No bonds found in topology. Try using '
                         'traj._topology.create_standard_bonds() to create bonds '
                         'using our PDB standard bond definitions.')
        
    nh_donors = get_donors('N', 'H')
    oh_donors = get_donors('O', 'H')
    #ch_donors = get_donors('C', 'H')
    #xh_donors = np.array(nh_donors + oh_donors + ch_donors)
    xh_donors = np.array(nh_donors + oh_donors)

    if len(xh_donors) == 0:
        # if there are no hydrogens or protein in the trajectory, we get
        # no possible pairs and return nothing
        return np.zeros((0, 3), dtype=int)

    acceptor_elements = frozenset(('O', 'N'))
    acceptors = [a.index for a in topology.atoms
        if a.element.symbol in acceptor_elements and can_participate(a, sur_resname)]
    
    
    # Make acceptors a 2-D numpy array
    acceptors = np.array(acceptors)[:, np.newaxis]

    # Generate the cartesian product of the donors and acceptors
    xh_donors_repeated = np.repeat(xh_donors, acceptors.shape[0], axis=0)
    acceptors_tiled = np.tile(acceptors, (xh_donors.shape[0], 1))
    bond_triplets = np.hstack((xh_donors_repeated, acceptors_tiled))

    # Filter out self-bonds
    self_bond_mask = (bond_triplets[:, 0] == bond_triplets[:, 2])
    return bond_triplets[np.logical_not(self_bond_mask), :]


def _compute_bounded_geometry(traj, triplets, distance_cutoff, distance_indices,
                              angle_indices, freq=0.0, periodic=True):
    """
    Returns a tuple include (1) the mask for triplets that fulfill the distance
    criteria frequently enough, (2) the actual distances calculated, and (3) the
    angles between the triplets specified by angle_indices.
    """
    # First we calculate the requested distances
    distances = compute_distances(traj, triplets[:, distance_indices], periodic=periodic)

    # Now we discover which triplets meet the distance cutoff often enough
    prevalence = np.mean(distances < distance_cutoff, axis=0)
    mask = prevalence > freq

    # Update data structures to ignore anything that isn't possible anymore
    triplets = triplets.compress(mask, axis=0)
    distances = distances.compress(mask, axis=1)

    # Calculate angles using the law of cosines
    abc_pairs = zip(angle_indices, angle_indices[1:] + angle_indices[:1])
    abc_distances = []

    # Calculate distances (if necessary)
    for abc_pair in abc_pairs:
        if set(abc_pair) == set(distance_indices):
            abc_distances.append(distances)
        else:
            abc_distances.append(compute_distances(traj, triplets[:, abc_pair],
                periodic=periodic))

    # Law of cosines calculation
    a, b, c = abc_distances
    cosines = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    np.clip(cosines, -1, 1, out=cosines) # avoid NaN error
    angles = np.arccos(cosines)

    return mask, distances, angles

def compute_neighbours (pdb, query_idxs, cutoff, water):
  """
  Determine which residues are within cutoff of query indices
  """
  neighbour_resnames=set()
  neighbour_atoms = md.compute_neighbors(pdb, cutoff, query_idxs)

  for atom_idx in neighbour_atoms[0]:
    resname =  pdb.topology.atom(atom_idx).residue
    neighbour_resnames.add(str(resname))

  return list(neighbour_resnames)

def _get_or_minus1(f):
    try:
        return f()
    except IndexError:
        return -1

