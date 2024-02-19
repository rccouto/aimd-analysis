#!/usr/bin/env python3

import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import re, optparse, copy, math
import numpy as np
import mdtraj as md


def get_tc_md_results(file):
    """ Read output from TeraChem MD calculation and get the CASSCF energy  
    
    Params: 
        file - output from Terachem (it works with combined outputs from
        multiple/restart files, i.e., simple 'cat run01.out run02.out > run-all.out')  
    
    Result/Return:
        ts - time step
        S - array with MD steps
        E - matrix with CASSCF energies with different roots in columms 
        D - transition dipole moments 
    """
    
    E=[]
    D=[]
    S=[]
    ts=0
    getEnergy=0

    dipole = re.compile(r'(\d+)\s+(\d+)\s+([+-]?\d+\.\d*(?:[Ee]-?\d+)?)\s+([+-]?\d+\.\d*(?:[Ee]-?\d+)?)')

    for i in open( file ).readlines():

        # Get the time step
        if re.search(r"Velocity Verlet integration time step:", i) is not None and ts == 0: 
            words = i.split()
            ts = float(words[5])
            #ts=1
           
        elif re.search(r"MD STEP", i) is not None and getEnergy == 0:
            words = i.split()
            s = int(words[4])

            if s == 1:
                S.append(s*ts)
                getEnergy=1
                #print("if")
            elif S and s > S[-1]:
                S.append(s*ts)
                getEnergy=1
            else:
                getEnergy=0
                #print("else")
                
        elif re.search(r"  singlet   ", i) is not None and getEnergy == 1 :
             words = i.split()
             e = float(words[2])
             E.append(e)
             #print(s)

        elif re.search(r"   1 ->  ", i) is not None and getEnergy == 1 :
             words = i.split()
             d = float(words[7])
             D.append(d)

        elif re.search(r"Singlet state velocity transition dipole moments:", i) is not None and getEnergy == 1:
             getEnergy=0

    nroots=int(len(E)/len(S))
    #print(nroots)
    E=np.array(E).reshape(-1,nroots)
    D=np.array(D).reshape(-1,2)

    return ts, S, E, D

def get_tc_sp_results(file):
    """ Read output from TeraChem MD calculation and get the CASSCF energy  
    
    Params: 
        file - output from Terachem (it works with combined outputs from
        multiple/restart files, i.e., simple 'cat run01.out run02.out > run-all.out')  
    
    Result/Return:
        E - matrix with CASSCF energies with different roots in columms 
        D - transition dipole moments 
    """
    
    D=[]
    E=[]
    for i in open( file ).readlines():
                
        if re.search(r"  singlet   ", i) is not None:
             words = i.split()
             e = float(words[2])
             E.append(e)
             #print(s)

        elif re.search(r"   1 ->  ", i) is not None:
             words = i.split()
             d = float(words[7])
             D.append(d)

        elif re.search(r"Singlet state velocity transition dipole moments:", i) is not None:
             break

    return E, D

def compute_rotation(
    xyz,
    A,
    B,
    C,
    D,
    ):

    r""" Compute the 360 rotation-angle property for geometry (in degrees).

    Params:
        xyz - the geometry
        A - the index of the first atom
        B - the index of the second atom
        C - the index of the third atom
        D - the index of the fourth atom
                                     
         \                /          
          \              /           
           \            /            
          C \__________/ B           
            /          \             
           /            \            
          /              \           
         D                A          

    Result/Return:
        theta - the float value of the torsion angle for the
            indices A, B, C, and D
    """

    
    rAB = xyz[B,:] - xyz[A,:]
    rBC = xyz[C,:] - xyz[B,:]
    rCD = xyz[D,:] - xyz[C,:]
    eAB = normalize(rAB)
    eBC = normalize(rBC)
    eCD = normalize(rCD)
    n1 = normalize(np.cross(rAB, rBC))
    n2 = normalize(np.cross(rBC, rCD))
    m1 = np.cross(n1, eBC)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    theta = 180 / math.pi * math.atan2(y, x)

    if theta < 0:
        theta = (theta*(-1)) + 180

    return theta

def normalize(vec):
    
    evec = copy.copy(vec)
    lvec = np.linalg.norm(evec)
    evec = evec/lvec

    return evec

def compute_neighbors (pdb, query_idxs, cutoff, water):
  """
  Determine which residues are within cutoff of query indices
  """
  
  neighbor_resnames=set()
  neighbor_resids = set()

  neighbor_atoms = md.compute_neighbors(pdb, cutoff, query_idxs)

  for atom_idx in neighbor_atoms[0]:
    resname =  pdb.topology.atom(atom_idx).residue
    resid =  pdb.topology.atom(atom_idx).residue.index
    if water == False: 
        if resid < 214:
            neighbor_resnames.add(resname)
            neighbor_resids.add(resid)
    else:
        neighbor_resnames.add(resname)
        neighbor_resids.add(resid)


  return list(neighbor_resids), list(neighbor_resnames)

def read_table(file):
    f = open(file, 'r')
    a = []
    for l in f.readlines():
        try:
            if not l.startswith('#'):
                a.append(l.split())
        except ValueError:
            pass
    f.close()
    return np.array(a)


# Set the Input Name
def OutName(name, ext, i):
    """" Setup a name of an output file in a formatted sequential way from a loop.
    
    Params: 
        name - the desired file name
        ext  - extension of desired file
        i    - the index

    Result/Return:
        File name in the proper format. 
    """

    if i > 9999:
        var = "-%s.%s" % (i,ext)
    elif i > 999:
        var = "-0%s.%s" % (i,ext)
    elif i > 99:
        var = "-00%s.%s" % (i,ext) 
    elif i > 9:
        var = "-000%s.%s" % (i,ext)       
    else:
        var = "-0000%s.%s" % (i,ext)
    
    out=name.rsplit( ".", 1 )[ 0 ] + var
    
    return out

def readxyz(filename):
    xyzf = open(filename, 'r')
    xyzarr = np.zeros([1, 3])
    atomnames = []
    if not xyzf.closed:
        # Read the first line to get the number of particles
        npart = int(xyzf.readline())
        # and next for title card
        title = xyzf.readline()

        # Make an N x 3 matrix of coordinates
        xyzarr = np.zeros([npart, 3])
        i = 0
        for line in xyzf:
            words = line.split()
            if (len(words) > 3):
                atomnames.append(words[0])
                xyzarr[i][0] = float(words[1])
                xyzarr[i][1] = float(words[2])
                xyzarr[i][2] = float(words[3])
                i = i + 1
    return (xyzarr, atomnames)


def rename_residue(name: str) -> str:
    """ MDTraj gives a 0-based index residue name. This routine takes a residue-atom full name and add 1 to the residue's index. 

    Args:
        name (str): 0-based residue name + atom

    Returns:
        new_name (str): 1-based residue name + atom
    """
    split_name=re.split(r'(\d+)',name)
    
    if len(split_name) > 3:
        new_name=split_name[0] + str((int(split_name[1])+1)) + split_name[2] + split_name[3]
    else:
        new_name=split_name[0] + str((int(split_name[1])+1)) + split_name[2]

    return new_name

def three_points_angle(left: list, center: list, right: list) -> float:
    """Compute the angle between three XYZ points. 

    Args:
        left (list): XYZ of point
        center (list): XYZ of point
        right (list): XYZ of point

    Returns:
        theta (float): angle in degrees
    """
    left_vec = normalize(left - center)
    right_vec = normalize(right - center)

    x = np.dot(left_vec, right_vec)
    theta = 180.0 / math.pi * np.arccos(x)

    return theta

# MAIN PROGRAM
def main():
    import sys
    f = optparse.OptionParser()
    # Get Type of Run
    f.add_option('-e', '--energy', type = str, default = None, help='Gets the CASSCF energy from TeraChem.')
    f.add_option('--dipole', type = str, default = None, help='Dipole moment ') 
    f.add_option('-o', '--out' , type = str, default = "bomd-all.out", help='TeraChem output file.')
    f.add_option('-a', '--analyze', type = str, default = None, help='Analyze trajectory from Terachem dynamics. Modules available: "hoop" (Pyramidalization angle) \n "rotation" (S-H rotation angle) \n "torsonly" (Save I- P-torsion to file)') 
    f.add_option('--hb' , action="store_true",  default=False, help='Monitor the Hydrogen bonding')
    f.add_option('--plot' , type = str,  default = None, help='Plot with matplotlib.')
    f.add_option('--closedist' , action="store_true",  default = False, help='Check the closest distance between residues and the chromophore.')
    f.add_option('--td' , action="store_true",  default = False, help='Flag for time-dependent analysis.')
    f.add_option('--name' , type = str, default = None, help='Name to be used in the output files/figures.')
    f.add_option('--spc' , type = str, default = None, help='Extract structures from trajectory for single point calculations. Module: run and get')
    f.add_option('--spcplot' ,action="store_true",  default=False, help='Extract structures from trajectory for single point calculations. Module: run and get')
    f.add_option('--test' , action="store_true",  default=False, help='For testing purposes')
    f.add_option('--h2o' , action="store_true",  default=False, help='For H20 hydrogen bonding only.')
    f.add_option('--surr' , action="store_true",  default=False, help='Gives a list with the close residues to the target in the whole trajectory.')
    f.add_option('--minima' , type=str,  default=None, help='Minima analysis: "meci", "is", "ls1d" ') 
    f.add_option('--meci' ,  type=int,  default=None, help='MECI analysis')
    f.add_option('--meci2' ,  type=str,  default=None, help='MECI analysis - all .dcd files in the folder')
    f.add_option('--meci3' ,  action="store_true",  default=False, help='MECI analysis - just for checking')
    f.add_option('--sim' , action="store_true",  default=False, help='Runs a similarities analysis.')
    f.add_option('--simclose' , action="store_true",  default=False, help='Runs a similarities analysis just for the close residues.')
    f.add_option('--torsion' , action="store_true",  default=False, help='Torsion analysis.')
    f.add_option('--violin' , action="store_true",  default=False, help='Make Violing plots')
    f.add_option('--violin2d' , action="store_true",  default=False, help='Make Violing plots from 2D US')
    f.add_option('--s1meci' , action="store_true",  default=False, help='Plot S1min and MECI structures together')
    f.add_option('--efield' , action="store_true",  default=False, help='Compute the electric field')
    f.add_option('--velscale' , action="store_true",  default=False, help='Scale velocity dcd file for hydrogen.')
    f.add_option('--dcd' , type=str,  default=None, help='Path for the dcd file to be read.')
    f.add_option('--dcd2' , type=str,  default=None, help='Path for the dcd file to be read.')
    f.add_option('--dcd3' , type=str,  default=None, help='Path for the dcd file to be read.')
    f.add_option('--dcdlist' , type=str,  default=None, help='File containing a dcd path list.')
    f.add_option('--top' , type=str,  default=None, help='Path for the prmtop file to be read.')
    f.add_option('--usdist' , type=str,  default=None, help='Plot the Umbrella Sampling distribution.')
    f.add_option('--com' , action="store_true",  default=False, help='Center of mass analysis')
    f.add_option('--file' , type=str,  default=None, help='Given file.')
    f.add_option('--projplane' , action="store_true",  default=False, help='Projection of atom positions on a plane')
    f.add_option('--distances' , action="store_true",  default = False, help='Plot distances.')
    f.add_option('--mmd' , action="store_true",  default = False, help='Analysis of MM production dynamics.')
    f.add_option('--distrib' , action="store_true",  default=False, help='Plot the distances distribution.')
    f.add_option('--mindist' , action="store_true",  default = False, help='Plot minimum distance between two residues in a trajectory.')


    (arg, args) = f.parse_args(sys.argv[1:])

    if len(sys.argv) == 1:
        f.print_help()
        sys.exit(1)

    # GET ENERGY MODULE
    if arg.energy:
        import numpy as np
        import matplotlib.pyplot as plt

        # GET STEP (fs) AND ENERGY (a.u.)
        st, step, energy, dipole = get_tc_md_results(arg.energy)
        
        # PLOT RESULS AND SAVE FIGURE
        plt.plot(step, (energy[:,1]-energy[:,0])*27.211385)
        plt.ylabel('Energy gap S1-S0 (eV)')
        plt.xlabel('Time (fs)')
        plt.title('Energy gap between S0 and S1')
        #plt.ylim(-80,80)
        plt.savefig('energy-gap-s0-s1.png')

        E=[]
        s0s1gap=(energy[:,1]-energy[:,0])*27.211385
        len(s0s1gap)
        for t in range(len(s0s1gap)):
            if (t % 4) == 0:
                E.append(s0s1gap[t])
                E.append(step[t])
        E=np.array(E).reshape(-1,2)
        #print(len(E))
        np.save("s0s1gap.npy", E)


        if arg.plot == "gap":
            plt.show()
        plt.close()

        # PLOT RESULS AND SAVE FIGURE
        plt.plot(step, energy[:,1])
        plt.ylabel('S1 energy (a.u.)')
        plt.xlabel('Time (fs)')
        plt.title('S1 energy')
        #plt.ylim(-80,80)
        plt.savefig('energy-s1.png')
        if arg.plot == "s1":
            plt.show()
        plt.close()

        # PLOT RESULS AND SAVE FIGURE
        plt.plot(step, energy[:,0])
        plt.ylabel('S0 energy (a.u.)')
        plt.xlabel('Time (fs)')
        plt.title('S0 energy')
        plt.savefig('energy-s0.png')
        if arg.plot == "s0":
            plt.show()
        plt.close()
        

    # ANALYZE THE TRAJECTORIES FROM TERACHEM
    elif arg.analyze:
        # Load MDTraj
        import mdtraj as md
        import numpy as np
        import socket
        import matplotlib.pyplot as plt

       
        # ON MACMINI    
        if socket.gethostname() == "rcc-mac.kemi.kth.se":
            sys.path.insert(1, '/Users/rafael/theochem/projects/codes/tcutil/code/geom_param') 
            import geom_param as gp
            # LOAD TRAJECTORIE(S)
            topology = md.load_prmtop('sphere.prmtop')
            traj = md.load_dcd('coors.dcd', top = topology)

        elif socket.gethostname() == "rcc-mac.kemi.kth.se" and arg.analyze == 'torsonly':
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
            import geom_param as gp
            topology = md.load_prmtop('sphere.prmtop')
            traj = md.load_dcd('coors-all.dcd', top = topology)
        
        elif socket.gethostname() == "nhlist-desktop":
            sys.path.insert(1, '/home/rcouto/theochem/progs/tcutil/code/geom_param') 
            import geom_param as gp
            # LOAD TRAJECTORIE(S)
            topology = md.load_prmtop(arg.top)
            if arg.dcd2 and arg.dcd :
                traj1 = md.load_dcd(arg.dcd, top = topology)
                traj2 = md.load_dcd(arg.dcd2, top = topology)
                traj=md.join([traj1,traj2], discard_overlapping_frames=True)
                del traj1, traj2
            else:
                traj = md.load_dcd(arg.dcd, top = topology)

        elif socket.gethostname() == "berzelius002" and arg.analyze == 'torsonly':
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
            import geom_param as gp
            topology = md.load_prmtop('sphere.prmtop')
            traj = md.load_dcd('coors-all.dcd', top = topology)

        elif socket.gethostname() == "berzelius002" and arg.analyze == 'all':
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
            import geom_param as gp
            topology = md.load_prmtop('sphere.prmtop')
            traj = md.load_dcd('prod.dcd', top = topology)
            #traj1 = md.load_dcd('scr.coors/coors.dcd', top = topology)
            #traj2 = md.load_dcd('res01/scr.coors/coors.dcd', top = topology)
            #traj=md.join([traj1,traj2], discard_overlapping_frames=True)
            #del traj1,traj2

        # ON BERZELIUS
        else:
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
            import geom_param as gp
            topology = md.load_prmtop('sphere.prmtop')
            traj1 = md.load_dcd('scr.coors/coors.dcd', top = topology)
            traj2 = md.load_dcd('res01/scr.coors/coors.dcd', top = topology)
            traj=md.join([traj1,traj2], discard_overlapping_frames=True)
            del traj1,traj2

        """
        elif  socket.gethostname() == "berzelius002":
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
            import geom_param as gp

            topology = md.load_prmtop('sphere.prmtop')
            traj1 = md.load_dcd('scr.coors/coors.dcd', top = topology)
            traj2 = md.load_dcd('res01/scr.coors/coors.dcd', top = topology)
            traj3 = md.load_dcd('res02/scr.coors/coors.dcd', top = topology)
            traj4 = md.load_dcd('res03/scr.coors/coors.dcd', top = topology)
            traj5 = md.load_dcd('res04/scr.coors/coors.dcd', top = topology)
            traj6 = md.load_dcd('res05/scr.coors/coors.dcd', top = topology)
            traj7 = md.load_dcd('res06/scr.coors/coors.dcd', top = topology)
            traj8 = md.load_dcd('res07/scr.coors/coors.dcd', top = topology)

            traj=md.join([traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8], discard_overlapping_frames=True)
            del traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8
        """


        # Chromophore indices
        chrome=[924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]

        # Radian to degrees
        r2d=57.2958

        if arg.analyze == "hoop" or arg.analyze == "all":
            """ 
            Compute and plot the teta angle for the whole trajectory
            """
            teta_pyra=[]
            for i in range(len(traj)):
                 # Pyra Indexes: 22(C4), 23(H8), 24(C1), 21(C5) - as in the List,ChemSci2022 paper
                teta = gp.compute_pyramidalization(traj.xyz[i,chrome,:],22,23,24,21)
                teta_pyra.append(teta)

            t=np.linspace(0, len(teta_pyra)-1, len(teta_pyra))*0.5
            plt.plot(t, teta_pyra)
            plt.ylabel('HOOP (deg)')
            plt.xlabel('Time (fs)')
            plt.ylim(-90,30)
            plt.title('HOOP')
            plt.savefig('hoop.png')
            plt.close()
            #plt.show(block = True)
            

        if arg.analyze == "rotation":
            """
            Compute and plot the rotation of angle of a group 
            """
            teta_rotation=[]
            for i in range(len(traj)):
                # S-H rotation angle
                teta = compute_rotation(traj.xyz[i,chrome,:],10,9,6,4)
                teta_rotation.append(teta)

            plt.plot(teta_rotation)
            plt.ylabel('Torsion (deg)')
            plt.xlabel('Timeframe')
            plt.title('S-H rotation')
            plt.savefig('sh-rotation.png')
            #plt.show(block = True)
            plt.close()

        if arg.analyze == "torsion":
            """"
            Compute the torsion 
            """
            import matplotlib.cm as cm

            # Compute the I- and P-torsion angles
            p_torsion=[]
            i_torsion=[]
            # Related atoms
            i_pair=[22,24]
            i_triple=[21,20,18]
            p_pair=[22,21]
            p_triple=[24,27,25]

            for i in range(len(traj)):
                # I-torsion
                teta = gp.compute_torsion5(traj.xyz[i,chrome,:],i_pair,i_triple)
                i_torsion.append(teta)
                # P-torsion
                teta = gp.compute_torsion5(traj.xyz[i,chrome,:],p_pair,p_triple)
                p_torsion.append(teta)

            # Save data to file
            np.save("i_torsion.npy",np.array(i_torsion))
            np.save("p_torsion.npy",np.array(p_torsion))


            Imax=np.nanmax(i_torsion)
            Imin=np.nanmin(i_torsion)
            for  count, t in enumerate(i_torsion):
                if t == Imax:
                    print("Itorsion max", count, t)
                    ImaxTraj=traj[count]
                    ImaxTraj.save_amberrst7('Imax.rst7')
                elif t == Imin:
                    print("Itorsion min", count, t)
                    IminTraj=traj[count]
                    IminTraj.save_amberrst7('Imin.rst7')

            Pmax=np.nanmax(p_torsion)
            Pmin=np.nanmin(p_torsion)
            for  count, t in enumerate(p_torsion):
                if t == Pmax:
                    print("Ptorsion max", count, t)
                    PmaxTraj=traj[count]
                    PmaxTraj.save_amberrst7('Pmax.rst7')
                elif t == Pmin:
                    print("Ptorsion min", count, t)
                    PminTraj=traj[count]
                    PminTraj.save_amberrst7('Pmin.rst7')

            moduleIP=abs(np.array(i_torsion))+abs(np.array(p_torsion))
            PImax=np.nanmax(moduleIP)
            for  count, t in enumerate(moduleIP):
                if t == PImax:
                    print("PItorsion max", count, t)
                    print("Itorsion", i_torsion[count])
                    print("Ptorsion", p_torsion[count])
                    PIminTraj=traj[count]
                    PIminTraj.save_amberrst7('PImax.rst7')
            

            # Time 
            t=np.linspace(0, len(i_torsion)-1, len(i_torsion))*0.5

            #  Plot I-torsion
            plt.plot(t,i_torsion)
            plt.ylabel('Dihedral angle (deg)')
            plt.xlabel('Time (fs)')
            plt.ylim(-80,80)
            plt.title('I Dihedral angle')
            plt.savefig('I_dihedral.png')
            #plt.show(block = True)
            plt.close()

            #  Plot P-torsion
            plt.plot(t,p_torsion)
            plt.ylabel('Dihedral angle (deg)')
            plt.xlabel('Time (fs)')
            plt.title('P Dihedral angle')
            plt.ylim(-80,80)
            plt.savefig('P_dihedral.png')
            #plt.show(block = True)
            plt.close()

            #  Plot I- and P-torsion together
            plt.plot(t,i_torsion, label="P-torsion")
            plt.plot(t,p_torsion, label="I-torsion")
            plt.ylabel('Dihedral angle (deg)')
            plt.xlabel('Time (fs)')
            plt.title('P- and I- Dihedral angle')
            plt.ylim(-80,80)
            plt.legend()
            plt.savefig('P-I_dihedral.png')
            #plt.show(block = True)
            plt.close()

            # 2D plot of I-torsion X P-torsion
            #n=len(i_torsion)
            #z=np.linspace(0, len(i_torsion)-1, len(i_torsion))
            alphas=np.linspace(1, 1, len(i_torsion))
            size=np.linspace(50, 50, len(i_torsion))

            #fig = plt.figure(figsize=(12, 12))
            #ax = fig.add_subplot(projection='3d')
            #ax.scatter(i_torsion,p_torsion,t, c=t, cmap=cmc.hawaii, s=size, alpha=alphas, linewidth=0.1)
            plt.scatter(i_torsion,p_torsion, c=t, cmap=cmc.hawaii, s=size, alpha=alphas, linewidth=0.1)
            plt.ylabel('P-torsion')
            plt.xlabel('I-torsion')
            plt.xlim(30,65)
            plt.ylim(-95,-50) 
            #plt.ylim(-43,30)
            plt.title('P-I-Torsion')
            cbar=plt.colorbar(values=t)
            cbar.set_label('Time (fs)')
            plt.savefig('i-p-torsion-2d.png')
            plt.close()
            #plt.show(block = True)

        if arg.analyze == "distance":
            
            if arg.file:
                table=read_table(arg.file)
            else:    
                table=read_table("closest-atoms-indexes.dat")
            #print(table)

            # Compute and plot the distance between two given atoms
            for i in range(len(table)):
                pair=np.array([[table[i,1],table[i,2]]], dtype=np.int32) # O P-ring <> Water353
                #print(pair)
                
                if arg.name:
                    fig_name = "%s-GYC-closest-dist-%s.png" % (table[i,0], arg.name)
                    data_name = "%s-GYC-closest-dist-%s.npy" % (table[i,0], arg.name)
                else:
                    fig_name = "%s-GYC-closest-dist.png" % (table[i,0])
                    data_name = "%s-GYC-closest-dist.npy" % (table[i,0])
                #print(fig_name)
                
                dist = md.compute_distances(traj,pair)
                # Save distance
                #np.save(data_name,np.array(dist))

                T=np.linspace(0,len(dist)/2,len(dist))

                plt.plot(T,dist*10)
                plt.ylabel('Distance (A)')
                plt.xlabel('Time (fs)')
                #plt.ylim(1.32,4.3) 
                if arg.name:
                    plt.title('Distance %s -- GYC (%s)' % (table[i,0], arg.name))
                else:
                    plt.title('Distance %s -- GYC' % table[i,0])
                plt.savefig(fig_name)
                #plt.show(block = True)
                plt.close()



        if arg.analyze == "flap" or arg.analyze == "all":
            """"
            Compute the flapping angle.
            """
            h1 = np.array([[944, 945, 948, 951]], dtype=np.int32)

            flap_dihedral_angle = md.compute_dihedrals(traj, h1)

            t=np.linspace(0, len(flap_dihedral_angle)-1, len(flap_dihedral_angle))*0.5

            plt.plot(t,flap_dihedral_angle*r2d)
            plt.ylabel('Flapping Dihedral angle (deg)')
            plt.xlabel('Time (fs)')
            plt.ylim(-25,5)
            plt.title('Flapping Dihedral angle')
            plt.savefig('flap_dihedral.png')
            #plt.show(block = True)
            plt.close()

        if arg.analyze == "torsonly" or arg.analyze == "all":
            """"
            Save I- P-torsion to file
            """
            import matplotlib.cm as cm

            # Compute the I- and P-torsion angles
            p_torsion=[]
            i_torsion=[]
            # Related atoms
            i_pair=[22,24]
            i_triple=[21,20,18]
            p_pair=[22,21]
            p_triple=[24,27,25]

            for i in range(len(traj)):
                # I-torsion
                teta = gp.compute_torsion5(traj.xyz[i,chrome,:],i_pair,i_triple)
                i_torsion.append(teta)
                # P-torsion
                teta = gp.compute_torsion5(traj.xyz[i,chrome,:],p_pair,p_triple)
                p_torsion.append(teta)

            # Save data to file
            np.save("i_torsion.npy",np.array(i_torsion))
            np.save("p_torsion.npy",np.array(p_torsion))

            # Time 
            t=np.linspace(0, len(i_torsion)-1, len(i_torsion))*0.5

            #  Plot I-torsion
            plt.plot(t,i_torsion)
            plt.ylabel('Dihedral angle (deg)')
            plt.xlabel('Time (fs)')
            plt.title('I Dihedral angle')
            plt.savefig('I_dihedral.png')
            plt.close()

            #  Plot P-torsion
            plt.plot(t,p_torsion)
            plt.ylabel('Dihedral angle (deg)')
            plt.xlabel('Time (fs)')
            plt.title('P Dihedral angle')
            plt.savefig('P_dihedral.png')
            plt.close()

            #  Plot I- and P-torsion together
            plt.plot(t,i_torsion, label="I-torsion")
            plt.plot(t,p_torsion, label="P-torsion")
            plt.ylabel('Dihedral angle (deg)')
            plt.xlabel('Time (fs)')
            plt.title('P- and I- Dihedral angle')
            plt.legend()
            plt.savefig('P-I_dihedral.png')
            plt.close()

            alphas=np.linspace(1, 1, len(i_torsion))
            size=np.linspace(50, 50, len(i_torsion))
            plt.scatter(i_torsion,p_torsion, c=t, cmap=cmc.hawaii, s=size, alpha=alphas, linewidth=0.1)
            plt.ylabel('P-torsion')
            plt.xlabel('I-torsion')
            plt.title('P-I-Torsion')
            cbar=plt.colorbar(values=t)
            cbar.set_label('Time (fs)')
            plt.savefig('P-I_dihedral-2D.png')
            plt.close()

            # Save last frame of trajectory
            LastFrame=traj[-1]
            LastFrame.save_amberrst7('LastFrame.rst7')

        if not arg.analyze:
            print("      Analyze module not available! \n      Rerun with -h to see the options.")
            sys.exit(1)

    

    if arg.closedist == True:
        import mdtraj as md
        import numpy as np
        import socket

        # LOAD TRAJECTORIE(S)
        if arg.top:
            topology = md.load_prmtop(arg.top)
        else:
            topology = md.load_prmtop('sphere.prmtop')            

        # ON MACMINI
        if socket.gethostname() == "rcc-mac.kemi.kth.se":
            if arg.dcd:
                traj = md.load_dcd(arg.dcd, top = topology)
            else:
                #traj = md.load_dcd('coors.dcd', top = topology)
                traj1 = md.load_dcd('f90_01.dcd', top = topology)
                traj2 = md.load_dcd('f90_02.dcd', top = topology)
                traj3 = md.load_dcd('f90_03.dcd', top = topology)
                traj=md.join([traj1,traj2,traj3], discard_overlapping_frames=True)
                del traj1,traj2,traj3

        else:
        # ON BERZELIUS
            if arg.dcd:
                traj = md.load_dcd(arg.dcd, top = topology)
            else:
                traj1 = md.load_dcd('scr.coors/coors.dcd', top = topology)
                traj2 = md.load_dcd('res01/scr.coors/coors.dcd', top = topology)
                traj3 = md.load_dcd('res02/scr.coors/coors.dcd', top = topology)
                traj4 = md.load_dcd('res03/scr.coors/coors.dcd', top = topology)
                traj5 = md.load_dcd('res04/scr.coors/coors.dcd', top = topology)
                traj6 = md.load_dcd('res05/scr.coors/coors.dcd', top = topology)
                traj7 = md.load_dcd('res06/scr.coors/coors.dcd', top = topology)
                traj8 = md.load_dcd('res07/scr.coors/coors.dcd', top = topology)

                traj=md.join([traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8], discard_overlapping_frames=True)
                del traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8


        Pring=['GYC60-OK', 'GYC60-FF', 'GYC60-FI', 'GYC60-FH', 'GYC60-HG']
        Bridge=['GYC60-HD']
        Iring=['GYC60-OA', 'GYC60-NB']

        if arg.td == True:
            chrome = traj.topology.select('resname GYC')

            out1 = open("closest-residues-Pring.dat", 'w')
            out2 = open("closest-residues-Bridge.dat", 'w')
            out3 = open("closest-residues-Iring.dat", 'w')
            out4 = open("closest-residues-All.dat", 'w')
            out5 = open("closest-residues-Summary.dat", 'w')

            out1.write("# Res_Name / Res_Atom_Id / GYC_Atom_Id\n")
            out2.write("# Res_Name / Res_Atom_Id / GYC_Atom_Id\n")
            out3.write("# Res_Name / Res_Atom_Id / GYC_Atom_Id\n")
            out4.write("# Res_Name / Res_Atom_Id / GYC_Atom_Id\n")

            out5.write("Closest atom from a residue to the chromophore\n")
            out5.write("Residue        Index   Chromophore     Index   Dist (AA)\n---------------------------------------------------------\n")
            
            try_set=set()
            # LOOP OVER EACH FRAME OF THE TRAJECTORY
            for f in range(len(traj)):
                trj=traj[f]
                sur_resids, sur_resname = compute_neighbors(trj,chrome,0.4, True)

                # LOOP OVER THE NEIGHBORS 
                for i in range(len(sur_resids)):
                    
                    if sur_resids[i] not in try_set:
                        try_set.add(sur_resids[i])

                        res_ids=trj.topology.select('resid %s' % sur_resids[i])

                        d=10000
                        # Run over the atoms within the residue
                        for atm in res_ids:
                            for chm in chrome:
                                pair=np.array([[atm,chm]], dtype=np.int32) 
                                dist = md.compute_distances(trj,pair)
                                
                                if dist < d:
                                    d = dist
                                    a1 = atm
                                    a2 = chm

                        resname= rename_residue(str(trj.topology.atom(a1)))
                        res_element=trj.topology.atom(a1).element
                        chrm_element=rename_residue(str(trj.topology.atom(a2)))


                        if d > 0.0 and str(chrm_element) in Pring:
                            out1.write("%s %d %d\n" % (resname, a1, a2))
                        elif d > 0.0 and str(chrm_element) in Bridge:
                            out2.write("%s %d %d\n" % (resname, a1, a2))
                        elif d > 0.0 and str(chrm_element) in Iring:
                            out3.write("%s %d %d\n" % (resname, a1, a2))
                        
                        out5.write('{:<8s}\t{:<5d}\t{:<8s}\t{:<4d}\t{:>2.2f}\n'.format(str(resname),  a1, str(chrm_element), a2, float(d*10)))
                        out4.write("%s %d %d\n" % (resname, a1, a2))

            out1.close()
            out2.close()
            out3.close()
            out4.close()
            out5.close()

        else:
            # Use just the first frame for this analyis
            traj=traj[-1]
            flag='final'

            #chrome = traj.topology.select('resname GYC')
            chrome = [940, 941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958]

            out1 = open(f"closest-residues-Pring_{flag}.dat", 'w')
            out2 = open(f"closest-residues-Bridge_{flag}.dat", 'w')
            out3 = open(f"closest-residues-Iring_{flag}.dat", 'w')
            out4 = open(f"closest-residues-All_{flag}.dat", 'w')
            out5 = open(f"closest-residues-Summary_{flag}.dat", 'w')
            out6 = open(f"index_dist_{flag}.dat", 'w')

            out1.write("# Res_Name / Res_Atom_Id / GYC_Atom_Id\n")
            out2.write("# Res_Name / Res_Atom_Id / GYC_Atom_Id\n")
            out3.write("# Res_Name / Res_Atom_Id / GYC_Atom_Id\n")
            out4.write("# Res_Name / Res_Atom_Id / GYC_Atom_Id\n")
            out5.write("Closest atom from a residue to the chromophore\n")
            out5.write("Residue        Index   Chromophore     Index   Dist (AA)\n---------------------------------------------------------\n")
            out6.write("# CONNECTION           ATOM 1  ATOM 2\n")

            try_set=set()
            # LOOP OVER EACH FRAME OF THE TRAJECTORY
   
            sur_resids, sur_resname = compute_neighbors(traj,chrome,0.5, True)

            # LOOP OVER THE NEIGHBORS 
            for i in range(len(sur_resids)):
                

                if sur_resids[i] not in try_set:
                    try_set.add(sur_resids[i])

                    res_ids=traj.topology.select('resid %s' % sur_resids[i])

                    d=10000
                    # Run over the atoms within the residue
                    for atm in res_ids:
                        for chm in chrome:
                            pair=np.array([[atm,chm]], dtype=np.int32) 
                            dist = md.compute_distances(traj,pair)
                            
                            if dist < d:
                                d = dist
                                a1 = atm
                                a2 = chm

                    resname= rename_residue(str(traj.topology.atom(a1)))
                    res_element=traj.topology.atom(a1).element
                    chrm_element=rename_residue(str(traj.topology.atom(a2)))

                    if d > 0.0 and str(chrm_element) in Pring:
                        out1.write("%s %d %d\n" % (resname, a1, a2))
                    elif d > 0.0 and str(chrm_element) in Bridge:
                        out2.write("%s %d %d\n" % (resname, a1, a2))
                    elif d > 0.0 and str(chrm_element) in Iring:
                        out3.write("%s %d %d\n" % (resname, a1, a2))
                    
                    out4.write("%s %d %d\n" % (resname, a1, a2))
                    out5.write('{:<8s}\t{:<5d}\t{:<8s}\t{:<4d}\t{:>2.2f}\n'.format(str(resname),  a1, str(chrm_element), a2, float(d*10)))
                    if resname != chrm_element:
                        out6.write('{:<7s}_{:<8s}\t{:<4d}\t{:<4d}\n'.format(resname,  str(chrm_element), a1, a2))

            out1.close()
            out2.close()
            out3.close()
            out4.close()
            out5.close()
            out6.close()

    if arg.hb == True:
        """" Identify the hydrogen bonds, that involves a given target, 
        present in the trajectory, for all frames of the dynamics (time-dependent).

        Params: the path to the trajectory files and the target should be given below.

        Result/Return:  It will save two files containing the time-depent information 
        regarding which atoms are involved in the HB network.
                        It will save a figure "hb.png" containing three plots:
                        1) Number of HB
                        2) Residue-Hydrogen and occurances. 
                        3) Residue-acceptor of the HB
                        All time-dependent  
        """

        import mdtraj as md
        import numpy as np
        import socket
        import matplotlib
        import matplotlib.colors
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        #################
        ## TARGET RESIDUE
        target='GYC60'
        chrome=[924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]
        #################

        # LOAD TRAJECTORIE(S)
        

        # ON MACMINI
        if socket.gethostname() == "rcc-mac.kemi.kth.se":
            import hbond as hb
            #traj = md.load_dcd('coors.dcd', top = topology)
            topology = md.load_prmtop('sphere.prmtop')
            traj = md.load_dcd('prod.dcd', top = topology)

        # ON BERZELIUS
        elif socket.gethostname() == "berzelius2.nsc.liu.se":
            sys.path.insert(1, '/proj/berzelius-2023-33/users/x_rafca/progs/tcutil/code/geom_param')
            import geom_param as gp
            import hbond as hb

            #topology = md.load_prmtop('sphere.prmtop')
            
            #traj1 = md.load_dcd('scr.coors/coors.dcd', top = topology)
            #traj2 = md.load_dcd('res01/scr.coors/coors.dcd', top = topology)
            #traj=md.join([traj1,traj2], discard_overlapping_frames=True)
            #del traj1,traj2

            # MD ANALYSIS
            topology = md.load_prmtop('pbc-box.prmtop')
            traj = md.load_dcd('prod.dcd', top = topology)

        else:
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
            import geom_param as gp
            sys.path.insert(1, '/proj/berzelius-2023-33/users/x_rafca/progs/aimd-analysis/')
            import hbond as hb

            topology = md.load_prmtop('sphere.prmtop')
            traj = md.load_dcd('coors-all.dcd', top = topology)



        """
        elif socket.gethostname() == "berzelius002":
        # ON BERZELIUS
            sys.path.insert(1, '/proj/berzelius-2023-33/users/x_rafca/progs/aimd-analysis/')
            import hbond as hb

            traj1 = md.load_dcd('scr.coors/coors.dcd', top = topology)
            traj2 = md.load_dcd('res01/scr.coors/coors.dcd', top = topology)
            traj3 = md.load_dcd('res02/scr.coors/coors.dcd', top = topology)
            traj4 = md.load_dcd('res03/scr.coors/coors.dcd', top = topology)
            traj5 = md.load_dcd('res04/scr.coors/coors.dcd', top = topology)
            traj6 = md.load_dcd('res05/scr.coors/coors.dcd', top = topology)
            traj7 = md.load_dcd('res06/scr.coors/coors.dcd', top = topology)
            traj8 = md.load_dcd('res07/scr.coors/coors.dcd', top = topology)

            print('-- Combining trajectories')
            traj=md.join([traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8], discard_overlapping_frames=True)
            del traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8
            print('-- DONE --')
        """


        # SET RELEVANT HB TO BE ANALYZED
        if arg.name == 'OK':
            relevant_hbs=['GYC60-OK']
            name='OK'
        elif arg.name == 'Iring':
            relevant_hbs=['GYC60-OA','GYC60-NB']
            name='Iring'
        else:
            relevant_hbs=['GYC60-OA','GYC60-OK','GYC60-NB','GYC60-HD','GYC60-HG','GYC60-HI','GYC60-HH','GYC60-HF']
            name = None

        # SET UP THE ARRAY FOR THE HB COUNT
        hb_size=100
        hb_hydrogen_count = np.zeros([(len(traj)), hb_size])
        hb_hydrogen_resname = np.empty([(len(traj)), hb_size], dtype = 'S10')
        hb_recept_count = np.zeros([(len(traj)), hb_size])
        hb_recept_resname = np.empty([(len(traj)), hb_size], dtype = 'S10')

        # SAVE INFO FILES 
        if name:
            out = open(f"hydrogen-bonding_{name}.dat", 'w')
            out2 = open(f"hydrogen-bonding-td_{name}.dat", 'w')
        else:
            out = open("hydrogen-bonding_all.dat", 'w')
            out2 = open("hydrogen-bonding-td_all.dat", 'w')

        out2.write("# t (fs) / N_HB /  RES  /  H   / CHROME\n")

        print('-- Computing HBs')
        # IDENTIFY THE HB WITH MODIFIED wernet_nilsson METHOD FROM MDTRAJ
        hbond = hb.wernet_nilsson(traj, target, exclude_water=False)
        print('-- DONE --')

        # SET UP ARRAYS AND SETS
        number_hb=[]
        hb_res_h=[]
        try_set=set()
        hb_recept=[]
        try_set_recept=set()
        
        

        # LOOP OF THE TRAJECTORY FRAMES
        for i in range(len(traj)):
            out.write("HBs \n")
            nhb = 0
            # LOOP OVER THE HYDROGEN BONDS LIST
            for hb in hbond[i]:
                # OPTIONS
                # Any HB involving GYC
                #if str(traj.topology.atom(hb[2]).residue) == target or  str(traj.topology.atom(hb[0]).residue) == target:

                # GYC60 only as HB receptor 
                #if str(traj.topology.atom(hb[2]).residue) == target:

                # HB involving relevant HB relevant_hbs
                if str(traj.topology.atom(hb[2])) in relevant_hbs or str(traj.topology.atom(hb[1])) in relevant_hbs:

                    # USE SET TO IDENTIFY UNIQUE HB RESIDUES-HYDROGEN AND APPEND TO ARRAY
                    if traj.topology.atom(hb[1]) not in try_set:
                        try_set.add(traj.topology.atom(hb[1]))
                        hb_res_h.append(str(traj.topology.atom(hb[1])))
                    
                    # LOOP OVER THE RESIDUE-HYDROGEN 
                    for j in range(len(hb_res_h)):
                        # IF RESIDUE ON THE LIST
                        if str(traj.topology.atom(hb[1])) == str(hb_res_h[j]):    
                            hb_hydrogen_count[i][j]+=1
                            hb_hydrogen_resname[i][j]=str(traj.topology.atom(hb[1]))
                            out.write("%d  %s -- %s -- %s \n" % (hb_hydrogen_count[i][j], traj.topology.atom(hb[0]), traj.topology.atom(hb[1]), traj.topology.atom(hb[2]) ) )
                            break
                    # HB COUNTER
                    nhb += 1 

                    # USE SET TO IDENTIFY UNIQUE HB RECEPTOR RESIDUES AND APPEND TO ARRAY
                    if traj.topology.atom(hb[2]) not in try_set_recept:
                        try_set_recept.add(traj.topology.atom(hb[2]))
                        hb_recept.append(str(traj.topology.atom(hb[2])))


                    # LOOP OVER RECEPTOR RESIDUES
                    for j in range(len(hb_recept)):
                        # IF residue in the list
                        if str(traj.topology.atom(hb[2])) == str(hb_recept[j]):    
                            hb_recept_count[i][j]+=1
                            hb_recept_resname[i][j]=str(traj.topology.atom(hb[2]))
                            break 

                    # WRITE TO FILE
                    out2.write("%6.1f %d %s --- %s --- %s \n" % (i*0.5, nhb, traj.topology.atom(hb[0]), traj.topology.atom(hb[1]), traj.topology.atom(hb[2]) ) )
            out.write("Time = %6.1f fs, Total HB %d \n" % (i*0.5, nhb))
            number_hb.append(nhb)

        out.write("Mean = %6.2f" % np.mean(number_hb))
        out2.write("Mean = %6.2f" % np.mean(number_hb))
        out.close()
        out2.close()

        # SETUP FIGURE 
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 2, 2]})
        fig.tight_layout()
        fig.set_figheight(12)
        fig.set_figwidth(18)
        plt.subplots_adjust(hspace=0)

        # PLOT THE HB COUNTER 
        # Time in ns
        t=np.linspace(0, len(hbond)-1, len(hbond))/100
        
        # FOR MD ANALYSIS
        #t=np.linspace(0, len(hbond)-1, len(hbond))


        ax1.bar(t,number_hb, width=0.1)
        ax1.set_ylabel("Number of HB")
        ax1.set_xticklabels([])

        if name:
            np.save(f"hb_number_{name}.npy", number_hb)
        else:
            np.save("hb_number_all.npy", number_hb)

        #custom_xlim=(0, len(hbond)*0.5)

        # FOR MD ANALYSIS
        custom_xlim=(0, len(hbond)/100)

        plt.setp((ax1, ax2, ax3), xlim=custom_xlim)

        ### PLOT THE RESIDUE-HYDROGEN COUNTER
        # RESIZE THE hb_hydrogen_count ARRAY TO hb_res_h
        hb_hydrogen_count=np.array(hb_hydrogen_count, order='F')
        hb_hydrogen_count.resize(len(traj), len(hb_res_h))

        # SET UP THE DATA
        r=np.linspace(0, len(hb_res_h)-1, len(hb_res_h))
        X,Y=np.meshgrid(t,r)
        Z=np.transpose(hb_hydrogen_count)

        #PLOT THE HB WHILE GROUPING SAME RESIDUES
        #cmap = matplotlib.colors.ListedColormap(['#000000','blue','red'])
        #ax2.set_yticks(r, list(hb_res_h))
        #ax2.set_ylim(-1, len(hb_res_h)+0.4)
        #ax2.set_ylabel("Residue-Hydrogen of the HB")
        #ax2.set_xlabel("Time (fs)")
        #ax2.scatter(X,Y,Z,c=Z, cmap=cmap, marker="o", alpha=1)

        # SPLIT THE LABELS
        HbH=[]
        for i in range(len(Z)): 
            maxHB=np.max(Z[i,:])
            for j in range(int(maxHB)):
                HbH.append(hb_res_h[i])
        
        # SPLIT THE HB COUNT
        HbHCount=[]
        HbHCount=np.zeros([len(HbH),len(traj)])
        
        RowCount=0
        # LOOP OVER NUMBER GROUPED HB RECEPTORS 
        for i in range(len(Z)): 
            maxHB=int(np.max(Z[i,:]))
            # LOOP OVER TIME
            for j in range(len(Z[1,:])):
                # LOOP NUMBER OF HB 
                for k in range(int(Z[i,j])):
                    HbHCount[RowCount+k][j]=10
            RowCount += maxHB
        
        # SET UP THE DATA
        #t=np.linspace(0, len(hbond)-1, len(hbond))*0.5
        r=np.linspace(0, len(HbH)-1, len(HbH))
        X,Y=np.meshgrid(t,r)
        Z=HbHCount

        # COLOR GROUPING
        colors=[]
        resCount=0
        residue='nothing'
        for res in HbH:
            if res == residue:
                colors.append(np.full((1, len(hbond)), resCount))
            else:
                resCount += 1
                colors.append(np.full((1, len(hbond)), resCount))
            residue=res

        #PLOT 
        cmap = matplotlib.colors.ListedColormap(['#000000','blue','red'])
        ax2.set_yticks(r, list(HbH))
        ax2.set_ylim(-1, len(HbH)+0.4)
        ax2.set_ylabel("Residue-Hydrogen of the HB")
        ax2.set_xlabel("Time (fs)")
        ax2.set_xticklabels([])
        ax2.scatter(X,Y,Z,c=colors, cmap='gist_rainbow', marker="o", alpha=1)

        if name:
            np.save(f"hb_donor_x_{name}.npy", X)
            np.save(f"hb_donor_y_{name}.npy", Y)
            np.save(f"hb_donor_z_{name}.npy", Z)
        else:
            np.save("hb_donor_x_all.npy", X)
            np.save("hb_donor_y_all.npy", Y)
            np.save("hb_donor_z_all.npy", Z)

        ### PLOT THE HB RECEPTOR 

        # RESIZE THE hb_recept_count ARRAY TO hb_recept
        hb_recept_count=np.array(hb_recept_count, order='F')
        hb_recept_count.resize(len(traj), len(hb_recept))
        
        # SET UP THE DATA
        #t=np.linspace(0, len(hbond)-1, len(hbond))*0.5
        # FOR MD ANALYSIS
        #t=np.linspace(0, len(hbond)-1, len(hbond))/100

        r=np.linspace(0, len(hb_recept)-1, len(hb_recept))
        X,Y=np.meshgrid(t,r)
        Z=np.transpose(hb_recept_count)

        #PLOT GROUPED
        #cmap = matplotlib.colors.ListedColormap(['blue','red','green'])
        #cmap = matplotlib.colors.ListedColormap(['#000000','blue','red','green', 'yellow'])
        #legend_elements = [ Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='1 HB'),
        #                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='2 HBs'),
        #                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green',label='3 HBs'),
        #                    Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',label='4 HBs')]

        #ax3.set_yticks(r, list(hb_recept))
        #ax3.legend(handles=legend_elements, ncol=4, loc="upper left")
        #ax3.set_ylabel("HB receptor")
        #ax3.set_xlabel("Time (fs)")
        #ax3.set_ylim(-1, len(hb_recept)+0.6)
        #ax3.scatter(X,Y,Z,c=Z, cmap=cmap, marker="o", alpha=1)
        
        # SPLIT THE LABELS
        HbRec=[]
        for i in range(len(Z)): 
            maxHB=np.max(Z[i,:])
            #print(i, Z[i,:])
            for j in range(int(maxHB)):
                HbRec.append(hb_recept[i])
        
        # SPLIT THE HB COUNT
        HbRecCount=[]
        HbRecCount=np.zeros([len(HbRec),len(traj)])

        RowCount=0
        # LOOP OVER NUMBER GROUPED HB RECEPTORS 
        for i in range(len(Z)): 
            maxHB=int(np.max(Z[i,:]))
            # LOOP OVER TIME
            for j in range(len(Z[0,:])):
                # LOOP NUMBER OF HB 
                for k in range(int(Z[i,j])):
                    HbRecCount[RowCount+k][j]=10
            RowCount += maxHB
        
        # SET UP THE DATA
        #t=np.linspace(0, len(hbond)-1, len(hbond))*0.5

        # FOR MD ANALYSIS
        #t=np.linspace(0, len(hbond)-1, len(hbond))/100
        
        r=np.linspace(0, len(HbRec)-1, len(HbRec))

        X,Y=np.meshgrid(t,r)
        Z=HbRecCount

        # COLOR GROUPING
        colors=[]
        resCount=0
        residue='nothing'
        for res in HbRec:
            if res == residue:
                colors.append(np.full((1, len(hbond)), resCount))
            else:
                resCount += 1
                colors.append(np.full((1, len(hbond)), resCount))
            residue=res

        #PLOT
        #cmap = matplotlib.colors.ListedColormap(['blue','red','green'])
        #cmap = matplotlib.colors.ListedColormap(['#000000','blue','red','green', 'yellow'])
        #legend_elements = [ Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='1 HB'),
        #                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='2 HBs'),
        #                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green',label='3 HBs'),
        #                    Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',label='4 HBs')]

        ax3.set_yticks(r, list(HbRec))
        #ax3.legend(handles=legend_elements, ncol=4, loc="upper left")
        ax3.set_ylabel("HB receptor")
        ax3.set_xlabel("Time (ns)")
        ax3.set_ylim(-1, len(HbRec)+0.6)
        ax3.scatter(X,Y,Z,c=colors, cmap='gist_rainbow', marker="o", alpha=1)
        
        if name:
            np.save(f"hb_acceptor_x_{name}.npy", X)
            np.save(f"hb_acceptor_y_{name}.npy", Y)
            np.save(f"hb_acceptor_z_{name}.npy", Z)
        else:
            np.save("hb_acceptor_x_all.npy", X)
            np.save("hb_acceptor_y_all.npy", Y)
            np.save("hb_acceptor_z_all.npy", Z)

        # SAVE FIGURE AND EXIT
        if name:
            plt.savefig(f'hb_{name}.png')
        else:
            plt.savefig('hb_all.png')
        #plt.show(block = True)
        plt.close()
        

    if arg.spc:
        """"
        Extract a given molecule from the a QM/MM AIMD trajectory 
        and to perform single point energy calculation with TeraChem. 
        """
        import mdtraj as md
        import numpy as np
        import socket, glob
        
        if arg.spc == 'run':
            # LOAD TRAJECTORIE(S)
            
            # ON MACMINI
            if socket.gethostname() == "rcc-mac.kemi.kth.se":
                topology = md.load_prmtop('sphere.prmtop')
                traj = md.load_dcd('coors.dcd', top = topology)
            # ON NHLIST-DESKTOP
            elif socket.gethostname() == "nhlist-desktop":
                topology = md.load_prmtop('/storage/nhlist/Jones_Dronpa2GFP/Dronpa2/8-aimd/WT/umbrella_sampling/1-sample-I/I90/sphere.prmtop')
                traj = md.load_dcd('/storage/nhlist/Jones_Dronpa2GFP/Dronpa2/8-aimd/WT/umbrella_sampling/1-sample-I/I90/scr/coors.dcd', top = topology)
            else:
                topology = md.load_prmtop('sphere.prmtop')
            # ON BERZELIUS
                traj1 = md.load_dcd('scr.coors/coors.dcd', top = topology)
                traj2 = md.load_dcd('res01/scr.coors/coors.dcd', top = topology)
                traj3 = md.load_dcd('res02/scr.coors/coors.dcd', top = topology)
                traj4 = md.load_dcd('res03/scr.coors/coors.dcd', top = topology)
                traj5 = md.load_dcd('res04/scr.coors/coors.dcd', top = topology)
                traj6 = md.load_dcd('res05/scr.coors/coors.dcd', top = topology)
                traj7 = md.load_dcd('res06/scr.coors/coors.dcd', top = topology)
                traj8 = md.load_dcd('res07/scr.coors/coors.dcd', top = topology)

                traj=md.join([traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8], discard_overlapping_frames=True)
                del traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8

            # Get the index for the target molecule.
            #target = traj.topology.select('resname GYC')
            target = [924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]
            # Index of connections to be substituted by H
            subH=[908,961]
            mol=np.append(target, subH)

            # Generate XYZ files for each trajectory frame. 
            for t in range(len(traj)):
                
                #if (t % 4) == 0:
                # File name
                xyz = OutName("tfhbdi", "xyz", t)
                out=open(xyz, 'w')

                out.write('{}\n'.format(len(mol)))
                out.write('Frame {}\n'.format(t))
                for i in range(len(mol)):
                    # Check if atom is in the H substitution list
                    if mol[i] in subH:
                        out.write('H\t{:>2.8f}\t{:>2.8f}\t{:>2.8f}\n'.format(traj.xyz[t,mol[i],0]*10,traj.xyz[t,mol[i],1]*10,traj.xyz[t,mol[i],2]*10))
                    else:
                        out.write('{}\t{:>2.8f}\t{:>2.8f}\t{:>2.8f}\n'.format(traj.topology.atom(mol[i]).element.symbol,traj.xyz[t,mol[i],0]*10,traj.xyz[t,mol[i],1]*10,traj.xyz[t,mol[i],2]*10))
                out.close()

        elif arg.spc == 'get':
            """
            Get the results from a single point calculation
            """
            import numpy as np

            t=[]
            e=[]
            d=[]
            outputs=sorted(glob.iglob('spc-f*.out'))
            for output in outputs:
                # GET STEP (fs) AND ENERGY (a.u.)
                energy, dipole = get_tc_sp_results(output)
                
                s1=(energy[1]-energy[0])*27.211385
                d1=dipole[0]

                time=int(output[5:10])/2
                t.append(time)
                e.append(s1)
                d.append(d1)

            Emd=np.load("s0s1gap.npy")
            Emd=np.array(Emd).reshape(-1,2)

            for i in range(len(e)):
                print(t[i], e[i])
                if i > 0 and e[i] == e[i-1]:
                    print("Same energy:", e[i], e[i-1])

            plt.plot(t,e, marker='.')
            plt.plot((Emd[:,1]-1)/2,Emd[:,0], marker='.', color='r')
            plt.ylabel('Osc.strength (a.u.)')
            plt.xlabel('Time (fs)')
            plt.title('S0->S1 Oscillator strength')
            plt.show()
                
            #out=arg.dipole
            #name=out.replace(".out", "-dipole.png")
            #plt.savefig(name, format='png', dpi=300)

        elif arg.spc == 'min':
            """ 
            Create xyz files for single point calculations from minimized structures
            """

            # Get the index for the target molecule.
            target = [924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]
            # Index of connections to be substituted by H
            subH=[908,961]
            mol=np.append(target, subH)


            files=sorted(glob.iglob('meci-*.dcd'))
            for file in files:

                prmtop=file.replace(".dcd", ".prmtop")
                topology = md.load_prmtop(prmtop)
                traj = md.load_dcd(file, top = topology)

                # WHICH FRAME TO USE
                #N=len(traj)-1
                N=0

                # File name
                xyz = file.replace(".dcd", ".xyz")
                xyz = xyz.replace("meci-", "")
                out=open(xyz, 'w')

                out.write('{}\n'.format(len(mol)))
                out.write('Frame {}\n'.format(N))
                for i in range(len(mol)):
                    # Check if atom is in the H substitution list
                    if mol[i] in subH:
                        out.write('H\t{:>2.8f}\t{:>2.8f}\t{:>2.8f}\n'.format(traj.xyz[N,mol[i],0]*10,traj.xyz[N,mol[i],1]*10,traj.xyz[N,mol[i],2]*10))
                    else:
                        out.write('{}\t{:>2.8f}\t{:>2.8f}\t{:>2.8f}\n'.format(traj.topology.atom(mol[i]).element.symbol,traj.xyz[N,mol[i],0]*10,traj.xyz[N,mol[i],1]*10,traj.xyz[N,mol[i],2]*10))
                out.close()


        else:
            print("Module | %s | not available in --spc." % arg.spc)
            sys.exit(1)


    if arg.surr == True:
        import mdtraj as md
        import socket
        
        # LOAD TRAJECTORIE(S)
        topology = md.load_prmtop('sphere.prmtop')

        # ON MACMINI
        if socket.gethostname() == "rcc-mac.kemi.kth.se":
            traj = md.load_dcd('coors.dcd', top = topology)
        else:
        # ON BERZELIUS
            traj1 = md.load_dcd('scr.coors/coors.dcd', top = topology)
            traj2 = md.load_dcd('res01/scr.coors/coors.dcd', top = topology)
            traj3 = md.load_dcd('res02/scr.coors/coors.dcd', top = topology)
            traj4 = md.load_dcd('res03/scr.coors/coors.dcd', top = topology)
            traj5 = md.load_dcd('res04/scr.coors/coors.dcd', top = topology)
            traj6 = md.load_dcd('res05/scr.coors/coors.dcd', top = topology)
            traj7 = md.load_dcd('res06/scr.coors/coors.dcd', top = topology)
            traj8 = md.load_dcd('res07/scr.coors/coors.dcd', top = topology)

            traj=md.join([traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8], discard_overlapping_frames=True)
            del traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8

        chrome = traj.topology.select('resname GYC')

        surr_res=set()
        for i in  range(len(traj)):
            print(i)
            sur_resids, sur_resname = compute_neighbors(traj[i],chrome,0.4, True)
            
            for sr in sur_resname:
                if sr not in surr_res:
                    surr_res.add(str(sr))
    
        print(list(surr_res))


    if arg.minima:
        import mdtraj as md 
        import socket
        import numpy as np
        import matplotlib.pyplot as plt


         # ON MACMINI    
        if socket.gethostname() == "rcc-mac.kemi.kth.se":
            sys.path.insert(1, '/Users/rafael/theochem/projects/codes/tcutil/code/geom_param') 
        # ON BERZELIUS
        else:
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
        import geom_param as gp

        if arg.minima == 'meci':
            dcd=['f2000-optim.dcd', 'f3400-optim.dcd', 'f5500-optim.dcd', 'f6900.21722-optim.dcd', 'f6900.837-optim.dcd', 'f7600-optim.dcd', 'f9000-optim.dcd']
            prm=['f2000-sphere.prmtop', 'f3400-sphere.prmtop', 'f5500-sphere.prmtop', 'f6900.21722-sphere.prmtop', 'f6900.837-sphere.prmtop', 'f7600-sphere.prmtop', 'f9000-sphere.prmtop']

            gasphase=['TFHBDI-MECII-acas.xyz', 'TFHBDI-MECIP-acas.xyz', 'TFHBDI-MECIP2-acas.xyz']

        elif arg.minima == 'is':
            dcd=['f2000-is-optim.dcd', 'f2700-is-optim.dcd', 'f3400-is-optim.dcd', 'f4100-is-optim.dcd', 'f4800-is-optim.dcd', 'f5500-is-optim.dcd', 'f6200-is-optim.dcd', 'f6900-is-optim.dcd', 'f7600-is-optim.dcd',  'f8300-is-optim.dcd', 'f9000-is-optim.dcd',]
            prm=['f2000-is-sphere.prmtop', 'f2700-is-sphere.prmtop', 'f3400-is-sphere.prmtop',  'f4100-is-sphere.prmtop', 'f4800-is-sphere.prmtop', 'f5500-is-sphere.prmtop', 'f6200-is-sphere.prmtop', 'f6900-is-sphere.prmtop', 'f7600-is-sphere.prmtop', 'f8300-is-sphere.prmtop', 'f9000-is-sphere.prmtop']

            gasphase=['TFHBDI-S1I2-acas.xyz', 'TFHBDI-S1planar-trans-acas.xyz', 'TFHBDI-S1P-acas.xyz', 'TFHBDI-S1I-acas.xyz', 'TFHBDI-S1planar-acas.xyz']

        elif arg.minima == 'ls1d':
            dcd=['f2000-ls1d-optim.dcd', 'f2700-ls1d-optim.dcd', 'f3400-ls1d-optim.dcd', 'f4100-ls1d-optim.dcd', 'f4800-ls1d-optim.dcd', 'f5500-ls1d-optim.dcd', 'f6200-ls1d-optim.dcd', 'f6900-ls1d-optim.dcd', 'f7600-ls1d-optim.dcd',  'f8300-ls1d-optim.dcd', 'f9000-ls1d-optim.dcd',]
            prm=['f2000-ls1d-sphere.prmtop', 'f2700-ls1d-sphere.prmtop', 'f3400-ls1d-sphere.prmtop',  'f4100-ls1d-sphere.prmtop', 'f4800-ls1d-sphere.prmtop', 'f5500-ls1d-sphere.prmtop', 'f6200-ls1d-sphere.prmtop', 'f6900-ls1d-sphere.prmtop', 'f7600-ls1d-sphere.prmtop', 'f8300-ls1d-sphere.prmtop', 'f9000-ls1d-sphere.prmtop']

            gasphase=['TFHBDI-S1I2-acas.xyz', 'TFHBDI-S1planar-trans-acas.xyz', 'TFHBDI-S1P-acas.xyz', 'TFHBDI-S1I-acas.xyz', 'TFHBDI-S1planar-acas.xyz']

        else:
            print("      Module not available! \n      Rerun with -h to see the options.")
            sys.exit(1)

        # Chromophore indices
        chrome=[924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]

        # Related atoms
        i_pair=[22,24]
        i_triple=[21,20,18]
        p_pair=[22,21]
        p_triple=[24,27,25]

        I=[]
        P=[]

        # READ OPTIMIZED DCD FILES
        for i in range(len(dcd)):
            topology = md.load_prmtop(prm[i])
            traj = md.load_dcd(dcd[i], top = topology)
            
            # I-torsion
            teta_i = gp.compute_torsion5(traj.xyz[0,chrome,:],i_pair,i_triple)
            # P-torsion
            teta_p = gp.compute_torsion5(traj.xyz[0,chrome,:],p_pair,p_triple)

            I.append(teta_i)
            P.append(teta_p)

        # Related atoms
        i_pair=[6,17]
        i_triple=[5,3,1]
        p_pair=[5,17]
        p_triple=[6,9,7]

        # READ XYZ STRUCTURES
        for i in range(len(gasphase)):
            coords, atoms = readxyz(gasphase[i])

            # I-torsion
            teta_i = gp.compute_torsion5(coords,i_pair,i_triple)
            # P-torsion
            teta_p = gp.compute_torsion5(coords,p_pair,p_triple)

            I.append(teta_i)
            P.append(teta_p)

        if arg.minima == "meci":
            label=['f2000', 'f3400', 'f5500', 'f6900.21722', 'f6900.837', 'f7600', 'f9000','GP-MECII', 'GP-MECIP', 'GP-MECIP2']
        else:
            label=['f2000', 'f2700', 'f3400', 'f4100', 'f4800', 'f5500', 'f6200', 'f6900', 'f7600', 'f8300', 'f9000', 'GP-S1I2', 'GP-S1planar-trans', 'GP-S1P', 'GP-S1I', 'GP-S1planar']

        fig, ax = plt.subplots()
        cmap = plt.get_cmap('jet', len(label))
        z=np.linspace(0,len(label)-1, len(label))
       
        scatter = ax.scatter(I,P, s=250, c=z, cmap=cmap, alpha=0.9)
        
        plt.legend(handles=scatter.legend_elements(num=len(label))[0], labels=label, loc='upper left', frameon=False)
        plt.xlabel("I torsion")
        plt.ylabel("P torsion")
        plt.xlim(-200,200)
        plt.ylim(-200,200) 

        if arg.minima == "meci":
            plt.title("MECI structures")
            plt.savefig('meci.png', dpi=300)
        elif arg.minima == "is":
            plt.title("S1 minimum from initial structure")
            plt.savefig('init-struct.png', dpi=300)
        elif arg.minima == "ls1d":
            plt.title("S1 minimum from lowest energy in dynamics")
            #plt.savefig('mim-dynam-struct.png', dpi=300)

        plt.show(block = True)
        plt.close()
        
    if arg.meci:
        import mdtraj as md 
        import socket
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        from matplotlib.lines import Line2D

         # ON MACMINI    
        if socket.gethostname() == "rcc-mac.kemi.kth.se":
            sys.path.insert(1, '/Users/rafael/theochem/projects/codes/tcutil/code/geom_param') 
        # ON BERZELIUS
        else:
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
        import geom_param as gp


        ## PLOT TYPE
        TYPE="pyr"
        #TYPE="ics"

        # Gas-phase MECI structures
        gasphase=['TFHBDI-MECII-acas.xyz', 'TFHBDI-MECIP-acas.xyz', 'TFHBDI-MECIP2-acas.xyz']

        #frame=[20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90]
        
        frame=[arg.meci]
        
        type1=['Imax', 'Imin', 'Pmax', 'Pmin', 'PImax']

        # Chromophore indices
        chrome=[924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]

        # Related atoms
        i_pair=[22,24]
        i_triple=[21,20,18]
        p_pair=[22,21]
        p_triple=[24,27,25]

        # PYRAMIDALIZATION INDEXES  
        pyr_idx= [22,23,24,21]

        # SETUP FIG
        fig, ax = plt.subplots()
        # COLORS
        if TYPE == "ics":
            color = cm.Paired(np.linspace(0, 1, len(type1)))
            markers=['o', 's', '^', '*', 'D']
        else:
            color = cm.Paired(np.linspace(0, 1, len(frame)))

        label=[]
        # READ OPTIMIZED DCD FILES
        for fm, c in zip(frame, color):
            I=[]
            P=[]
            PYR=[]
            for tp, c2, mk in zip(type1, color, markers):

                topology = md.load_prmtop(f'meci-f{fm}-{tp}.prmtop')
                traj = md.load_dcd(f'meci-f{fm}-{tp}.dcd', top = topology)
                N=len(traj)-1
                # I-torsion
                teta_i = gp.compute_torsion5(traj.xyz[N,chrome,:],i_pair,i_triple)
                # P-torsion
                teta_p = gp.compute_torsion5(traj.xyz[N,chrome,:],p_pair,p_triple)
                # Pyramizadlization
                teta_pyr = gp.compute_pyramidalization(traj.xyz[N,chrome,:],pyr_idx[0],pyr_idx[1],pyr_idx[2],pyr_idx[3])
                
                I.append(teta_i)
                P.append(teta_p)
                PYR.append(teta_pyr)

                if TYPE == "ics":
                    # Initial geometry
                    Iteta_i = gp.compute_torsion5(traj.xyz[0,chrome,:],i_pair,i_triple)
                    Iteta_p = gp.compute_torsion5(traj.xyz[0,chrome,:],p_pair,p_triple)
                    Iteta_pyr = gp.compute_pyramidalization(traj.xyz[0,chrome,:],pyr_idx[0],pyr_idx[1],pyr_idx[2],pyr_idx[3])
                    
                    scatter = ax.scatter(Iteta_i,Iteta_p, s=150, c=Iteta_pyr, cmap='coolwarm', alpha=0.7, edgecolors='black', marker=mk, linewidths=1, vmin=-40, vmax=40)

                    # Final geometry
                    scatter = ax.scatter(teta_i,teta_p, s=150,  c=teta_pyr, cmap='coolwarm', alpha=0.7, marker=mk, vmin=-40, vmax=40)

                    # Connecting line
                    ax.plot([teta_i, Iteta_i], [teta_p, Iteta_p], ls="-", c=".1", alpha=0.5)
                    #plt.arrow(x=Iteta_i, y=Iteta_p, dx=(teta_i-Iteta_i), dy=(teta_p-Iteta_p), width=0.8,facecolor='black', edgecolor='none')

                #print("f%d00 \t %s \t %3.2f \t %3.2f" % (fm, tp, teta_i, teta_p))
            label.append(str(f'f{fm}00'))

            if TYPE == "pyr":
                scatter = ax.scatter(I,P, s=150, c=PYR, cmap='coolwarm', alpha=0.8, vmin=-40, vmax=40)
           
        if TYPE == "ics":
            cmap = plt.cm.coolwarm
            legend_elements = [Line2D([0], [0], color='w', markerfacecolor=cmap(0.), label=type1[0], marker=markers[0], markersize=10),
                            Line2D([0], [0], color='w', markerfacecolor=cmap(0.), label=type1[1], marker=markers[1], markersize=10),
                            Line2D([0], [0], color='w', markerfacecolor=cmap(0.), label=type1[2], marker=markers[2], markersize=10),
                            Line2D([0], [0], color='w', markerfacecolor=cmap(0.), label=type1[3], marker=markers[3], markersize=15),
                            Line2D([0], [0], color='w', markerfacecolor=cmap(0.), label=type1[4], marker=markers[4], markersize=10),
                            Line2D([0], [0], color='w', markeredgecolor='black', markerfacecolor='w', label='Init. geom.', marker='s', markersize=10)
                            ]
            plt.legend(handles=legend_elements, frameon=False)
            #plt.legend(hanfles=markers, loc='upper right', frameon=False)
            plt.title(f"IC f{fm}00")

        
        cbar=plt.colorbar(scatter)
        cbar.set_label(r'$\theta_{pyr}$ (degrees)',fontsize=14)

        # Related atoms
        i_pair=[6,17]
        i_triple=[5,3,1]
        p_pair=[5,17]
        p_triple=[6,9,7]

        Igas=[]
        Pgas=[]
        # READ XYZ STRUCTURES
        for i in range(len(gasphase)):
            coords, atoms = readxyz(gasphase[i])

            # I-torsion
            teta_i = gp.compute_torsion5(coords,i_pair,i_triple)
            # P-torsion
            teta_p = gp.compute_torsion5(coords,p_pair,p_triple)
            Igas.append(teta_i)
            Pgas.append(teta_p)
            LabelName=gasphase[i]
            label.append(f"GP-{LabelName[7:13]}")

        ax.plot([200, -200], [-200, 200], ls="--", c=".1", alpha=0.5)
        plt.xlabel(r"$\phi_I$ (degrees)",fontsize=14)
        plt.ylabel(r"$\phi_P$ (degrees)",fontsize=14)
        
        if TYPE == "ics":
            #plt.xlim(-80,90)
            #plt.ylim(-100,80)
            plt.xlim(20,70)
            plt.ylim(-100,-40)
        else:
            plt.xlim(-120,120)
            plt.ylim(-120,120)

        scatter = ax.scatter(Igas,Pgas, s=100, color='red', marker='D')
        
        #plt.title("MECI structures")
        #plt.savefig('meci-opt.svg', dpi=300, format='svg')

        if TYPE == "ics":
            out=f"meci-opt-f{fm}00-init-ZOOM.png"
            plt.savefig(out, dpi=300, format='png')
        #else:


        plt.show(block = True)
        #plt.close()
        

    if arg.meci2:
        import mdtraj as md 
        import socket, glob
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        from matplotlib.lines import Line2D

         # ON MACMINI    
        if socket.gethostname() == "rcc-mac.kemi.kth.se":
            sys.path.insert(1, '/Users/rafael/theochem/projects/codes/tcutil/code/geom_param') 
        # ON BERZELIUS
        else:
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
        import geom_param as gp

        if arg.meci2 == 's1':
            gasphase=['TFHBDI-S1I2-acas.xyz', 'TFHBDI-S1planar-trans-acas.xyz', 'TFHBDI-S1P-acas.xyz', 'TFHBDI-S1I-acas.xyz', 'TFHBDI-S1planar-acas.xyz']
        else:
            # Gas-phase MECI structures
            gasphase=['TFHBDI-MECII-acas.xyz', 'TFHBDI-MECIP-acas.xyz', 'TFHBDI-MECIP2-acas.xyz']

        # Chromophore indices
        chrome=[924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]

        # Related atoms
        i_pair=[22,24]
        i_triple=[21,20,18]
        p_pair=[22,21]
        p_triple=[24,27,25]

        # PYRAMIDALIZATION INDEXES  
        pyr_idx= [22,23,24,21]

        # SETUP FIG
        fig, ax = plt.subplots()
        
        if arg.meci2 == 's1':
            files=sorted(glob.iglob('*.dcd'))
        else:
            files=sorted(glob.iglob('meci-f90*.dcd'))

        # COLORS
        color = cm.Paired(np.linspace(0, 1, len(files)))

        cmap = plt.get_cmap('jet', 11)
        z=np.linspace(0,10, 11)

        lbl=np.zeros(11)

        label=[]
        I=[]
        P=[]
        PYR=[]
        for file, c in zip(files, color):

            prmtop=file.replace(".dcd", ".prmtop")

            topology = md.load_prmtop(prmtop)
            traj = md.load_dcd(file, top = topology)
            N=len(traj)-1
            #N=0
            # I-torsion
            teta_i = gp.compute_torsion5(traj.xyz[N,chrome,:],i_pair,i_triple)
            # P-torsion
            teta_p = gp.compute_torsion5(traj.xyz[N,chrome,:],p_pair,p_triple)
            # Pyramizadlization
            teta_pyr = gp.compute_pyramidalization(traj.xyz[N,chrome,:],pyr_idx[0],pyr_idx[1],pyr_idx[2],pyr_idx[3])
            
            # Final geometry
            #scatter = ax.scatter(teta_i,teta_p, s=70,  c=teta_pyr, cmap='coolwarm', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)

            if "f20" in file:
                if lbl[0] == 0:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='blue', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40, label="f20")
                    lbl[0]=1
                else:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='blue', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)
                
            elif "f27" in file:
                if lbl[1] == 0:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='red', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40,  label="f27")
                    lbl[1]=1
                else:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='red', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)

            elif "f34" in file:
                if lbl[2] == 0:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='green', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40,  label="f34")
                    lbl[2]=1
                else:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='green', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)
            elif "f41" in file:
                if lbl[3] == 0:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='black', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40,  label="f41")
                    lbl[3]=1
                else:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='black', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)
            elif "f48" in file:
                if lbl[4] == 0:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='orange', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40,  label="f48")
                    lbl[4]=1
                else:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='orange', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)
            elif "f55" in file:
                if lbl[5] == 0:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='yellow', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40,  label="f55")
                    lbl[5]=1
                else:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='yellow', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)
            elif "f62" in file:
                if lbl[6] == 0:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='lime', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40,  label="f62")
                    lbl[6]=1
                else:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='lime', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)

            elif "f69" in file:
                if lbl[7] == 0:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='pink', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40,  label="f69")
                    lbl[7]=1
                else:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='pink', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)
            elif "f76" in file:
                if lbl[8] == 0:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='purple', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40,  label="f76")
                    lbl[8]=1
                else:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='purple', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)
            elif "f83" in file:
                if lbl[9] == 0:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='olive', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40,  label="f83")
                    lbl[9]=1
                else:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='olive', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)
            elif "f90" in file:
                if lbl[10] == 0:
                    scatter = ax.scatter(teta_i,teta_p, s=100,  c='navy', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40,  label="f90")
                    lbl[10]=1
                else:
                    scatter = ax.scatter(teta_i,teta_p, s=70,  c='navy', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)
            #else: 
            #    scatter = ax.scatter(teta_i,teta_p, s=70,  c='green', alpha=0.7,  edgecolors='black', linewidths=0.5, vmin=-40, vmax=40)

            #colors=['black', 'silver', 'red',  'salmon', 'brown', 'orange', 'gold','yellow', 'olive', 'green', 'lime', 'teal', 'aqua', 'blue', 'navy', 'violet', 'lavender', 'magenta', 'pink', 'purple', 'black']

            I.append(teta_i)
            P.append(teta_p)
            PYR.append(teta_pyr)

            # Initial geometry
            #Iteta_i = gp.compute_torsion5(traj.xyz[0,chrome,:],i_pair,i_triple)
            #Iteta_p = gp.compute_torsion5(traj.xyz[0,chrome,:],p_pair,p_triple)
            #Iteta_pyr = gp.compute_pyramidalization(traj.xyz[0,chrome,:],pyr_idx[0],pyr_idx[1],pyr_idx[2],pyr_idx[3])       
            #scatter = ax.scatter(Iteta_i,Iteta_p, s=50, c=Iteta_pyr, cmap='coolwarm', alpha=0.9, vmin=-40, vmax=40)

            #if len(file) == 16: 
            #    name=file[5:12]
            #elif len(file) == 17:
            #    name=file[5:13]
            #else:
            #    name=file[5:14]
            name=file[5:8]
            #ax.annotate(name, (Iteta_i,Iteta_p), fontsize=6)

            # Connecting line
            #ax.plot([teta_i, Iteta_i], [teta_p, Iteta_p], ls="--", c=".001", alpha=0.3)

        #scatter = ax.scatter(I,P, s=150, c=PYR, cmap='coolwarm', alpha=0.8, vmin=-40, vmax=40)
        
        #cbar=plt.colorbar(scatter)
        #cbar.set_label(r'$\theta_{pyr}$ (degrees)',fontsize=14)

        # Related atoms
        i_pair=[6,17]
        i_triple=[5,3,1]
        p_pair=[5,17]
        p_triple=[6,9,7]

        Igas=[]
        Pgas=[]
        # READ XYZ STRUCTURES
        #for i in range(len(gasphase)):
        #    coords, atoms = readxyz(gasphase[i])
        #    # I-torsion
        #    teta_i = gp.compute_torsion5(coords,i_pair,i_triple)
        #    # P-torsion
        #    teta_p = gp.compute_torsion5(coords,p_pair,p_triple)
        #    Igas.append(teta_i)
        #    Pgas.append(teta_p)
        #    LabelName=gasphase[i]
        #    label.append(f"GP-{LabelName[7:13]}")
        #scatter = ax.scatter(Igas,Pgas, s=70, color='red', marker='D')

        ax.plot([200, -200], [-200, 200], ls="-", c=".1", alpha=0.2)
        ax.plot([-130,130], [0,0], ls="-", c=".1", alpha=0.1)
        ax.plot( [0,0], [-130,130],ls="-", c=".1", alpha=0.1)
        ax.plot( [90,90], [-130,130],ls=":", c=".1", alpha=0.1)
        ax.plot( [-90,-90], [-130,130],ls=":", c=".1", alpha=0.1)
        ax.plot(  [-130,130],[90,90],ls=":", c=".1", alpha=0.1)
        ax.plot(  [-130,130],[-90,-90],ls=":", c=".1", alpha=0.1)

        plt.xlabel(r"$\phi_I$ (degrees)",fontsize=14)
        plt.ylabel(r"$\phi_P$ (degrees)",fontsize=14)
        plt.xlim(-120,120)
        plt.ylim(-120,120)
        plt.xticks(np.arange(-120,121,step=30))
        plt.yticks(np.arange(-120,121,step=30))
        plt.legend()

        
        
        plt.title(f"MECI structures - {name}")
        
        plt.savefig(f'MECI-dyn-{name}.png', dpi=300, format='png')

        #plt.show(block = True)
        plt.close()


    if arg.s1meci:
        import mdtraj as md 
        import socket, glob
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        from matplotlib.lines import Line2D

         # ON MACMINI    
        if socket.gethostname() == "rcc-mac.kemi.kth.se":
            sys.path.insert(1, '/Users/rafael/theochem/projects/codes/tcutil/code/geom_param') 
        # ON BERZELIUS
        else:
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
        import geom_param as gp

        # Chromophore indices
        chrome=[924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]

        # TORSION INDEXES
        i_pair=[22,24]
        i_triple=[21,20,18]
        p_pair=[22,21]
        p_triple=[24,27,25]

        # PYRAMIDALIZATION INDEXES  
        pyr_idx= [22,23,24,21]

        # SETUP FIG
        #fig, ax = plt.subplots()
        
        s1files=sorted(glob.iglob('s1min*.dcd'))
        # COLORS
        color = cm.Paired(np.linspace(0, 1, len(s1files)))
        
        for s1file, c in zip(s1files, color):
            # SETUP FIG
            fig, ax = plt.subplots()

            # S1 prmtop
            s1prmtop=s1file.replace(".dcd", ".prmtop")

            # MECI FILES
            mecifile=s1file.replace("s1min", "meci")
            meciprmtop=mecifile.replace(".dcd", ".prmtop")

            # LOAD S1 TRAJECTORY
            S1top = md.load_prmtop(s1prmtop)
            S1traj = md.load_dcd(s1file, top = S1top)

            # LOAD MECI TRAJECTORY
            MECItop = md.load_prmtop(meciprmtop)
            MECItraj = md.load_dcd(mecifile, top = MECItop)


            # INITIAL GEOMETRY
            Iteta_i = gp.compute_torsion5(S1traj.xyz[0,chrome,:],i_pair,i_triple)
            Iteta_p = gp.compute_torsion5(S1traj.xyz[0,chrome,:],p_pair,p_triple)
            Iteta_pyr = gp.compute_pyramidalization(S1traj.xyz[0,chrome,:],pyr_idx[0],pyr_idx[1],pyr_idx[2],pyr_idx[3]) 
            scatter = ax.scatter(Iteta_i,Iteta_p, s=100, c=Iteta_pyr, cmap='coolwarm', alpha=1, vmin=-40, vmax=40,  edgecolors='black', linewidths=1)

            # FINAL S1min
            S1teta_i = gp.compute_torsion5(S1traj.xyz[-1,chrome,:],i_pair,i_triple)
            S1teta_p = gp.compute_torsion5(S1traj.xyz[-1,chrome,:],p_pair,p_triple)
            S1teta_pyr = gp.compute_pyramidalization(S1traj.xyz[-1,chrome,:],pyr_idx[0],pyr_idx[1],pyr_idx[2],pyr_idx[3])
            scatter = ax.scatter(S1teta_i,S1teta_p, s=100, c=S1teta_pyr, cmap='coolwarm', alpha=1, vmin=-40, vmax=40)

            # FINAL MECI
            MECIteta_i   = gp.compute_torsion5(MECItraj.xyz[-1,chrome,:],i_pair,i_triple)
            MECIteta_p   = gp.compute_torsion5(MECItraj.xyz[-1,chrome,:],p_pair,p_triple)
            MECIteta_pyr = gp.compute_pyramidalization(MECItraj.xyz[-1,chrome,:],pyr_idx[0],pyr_idx[1],pyr_idx[2],pyr_idx[3])
            scatter = ax.scatter(MECIteta_i,MECIteta_p, s=100, c=MECIteta_pyr, cmap='coolwarm', alpha=1, vmin=-40, vmax=40)
            #scatter = ax.scatter(MECIteta_i,MECIteta_p, s=10, color='g', alpha=0.9)
            
            # LABEL
            if len(s1file) == 16: 
                name=s1file[6:12]
            elif len(s1file) == 17:
                name=s1file[6:13]
            else:
                name=s1file[6:14]
            #ax.annotate(name, (Iteta_i,Iteta_p), fontsize=6)

            ax.annotate("Init",  (Iteta_i+1,Iteta_p), fontsize=8)
            ax.annotate("S1min", (S1teta_i,S1teta_p), fontsize=8)
            ax.annotate("MECI",  (MECIteta_i,MECIteta_p), fontsize=8)

            # CONNECTING LINE INIT -> S1
            ax.plot([Iteta_i, S1teta_i], [Iteta_p, S1teta_p], ls="--", lw="0.8", c=".01", alpha=0.1)
            # CONNECTING LINE INIT -> S1
            ax.plot([Iteta_i, MECIteta_i], [Iteta_p, MECIteta_p], ls="--", lw="0.8", c=".01", alpha=0.1)
            # CONNECTING LINE S1 -> MECI
            ax.plot([S1teta_i, MECIteta_i], [S1teta_p, MECIteta_p],  ls="--", c="r", alpha=0.5)
        
            #cbar=plt.colorbar(scatter)
            #cbar.set_label(r'$\theta_{pyr}$ (degrees)',fontsize=14)
            plt.title(name)
            plt.xlabel(r"$\phi_I$ (degrees)",fontsize=14)
            plt.ylabel(r"$\phi_P$ (degrees)",fontsize=14)
            cbar=plt.colorbar(scatter)
            cbar.set_label(r'$\theta_{pyr}$ (degrees)',fontsize=14)

            plt.savefig(f's1min-meci-{name}.png', dpi=300, format='png')
            #plt.show(block = True)
            plt.close()

        
        #plt.savefig(f's1-meci-ALL.png', dpi=300, format='png')
        #plt.show(block = True)

    if arg.sim == True:    
        """
        Similaritites module

        """
        import mdtraj as md
        import numpy as np
        import matplotlib

        frames=['f2000', 'f2700', 'f3400', 'f4100', 'f4800', 'f5500', 'f6200', 'f6900', 'f7600', 'f8300', 'f9000']
        #frames=['f2000']
        
        # SIDECHAIN
        #HalfLine=[0.9442111401289468, 0.8771515044856577, 2.56018847039622, 1.5266326665124563, 0.7462042190247257, 0.8469578774597802, 0.9615571258905131, 1.0387622352852939, 1.4527264375858053, 1.8736762535021434, 1.72978352715429]
        # BACKBONE
        HalfLine=[-0.17747473034382233, -0.11045843799696559, -0.2064068263802431, -0.15079907558004726, -0.07003044101252691, -0.09834329160200583, -0.1838430207833931, -0.12693914729423988, -0.3225691366605811, -0.1442561473453926, -0.13137948671085134]

        for count, frame in enumerate(frames):
            print("\n", frame)
            # Load PDB
            pdb = md.load_pdb(f"{frame}-1stFrame.pdb")
            chrome = pdb.topology.select('resname GYC')

            # Load similatities results
            simList=np.load(f"../results_sidechain/{frame}_KLD_sidechain.npy")
            simListBackbone=np.load(f"../results_backbone/{frame}_KLD_backbone.npy")

            simList=simList-simListBackbone

            simRes=np.linspace(0, len(simList)-1, len(simList))
            
            dist = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.4]

            colors=['black', 'red', 'green', 'blue', 'cyan', 'pink', 'gray', 'orange']
            
            fig, axs=plt.subplots(4, 2)
            fig.subplots_adjust(hspace = 0.2, wspace=0.1)
            fig.set_figheight(12)
            fig.set_figwidth(20)
            axs = axs.ravel()

            prevRes = [99999]
            j=0
            for cutoff in dist:
                #print("\n ", cutoff)
                resids, resnames = compute_neighbors(pdb,chrome,cutoff, False)
                resids.sort()

                currentRes=np.array(resids)
                layer=np.setdiff1d(currentRes, prevRes)

                layerRes=[]
                layerTest=[]
                for i in range(len(simList)):
                    if simRes[i] in layer:
                        layerTest.append(int(simRes[i]+1))
                        layerRes.append(simList[i])
                        #if simList[i] > HalfLine[count]:
                        #    print( int(simRes[i]+1), end =" ")

                #print(np.amin(layerRes)/2)
                #print(np.max(layerRes)/2)
                #HalfLine=1.8884222802578936/2

                x = np.arange(len(layerTest))
                wid=len(x)/70
                axs[j].bar(x,layerRes, color=colors[j], width=wid)
                axs[j].xaxis.set_tick_params(rotation=45)
                axs[j].set_xticks(x, layerTest, fontsize=9)
                axs[j].axhline(y=HalfLine[count],linewidth=1, color='k', linestyle="--")

                #if frame == "f3400":
                #    axs[j].set_ylim(0,5.2)
                #elif frame == "f7600":
                #    axs[j].set_ylim(0,3)
                #elif frame == "f6900":
                #    axs[j].set_ylim(0,2.5)
                #elif frame == "f8300" or frame == "f9000" or frame == "f4100":
                #    axs[j].set_ylim(0,4)
                #else:
                #    axs[j].set_ylim(0,2)

                yMax=(HalfLine[count]*2)-0.05
                axs[j].set_ylim(0,yMax)


                if j == 0:
                    axs[j].set_title(r"Radius < %2.1f $\AA$" % (dist[j]*10), y=1.0, pad=-14)
                elif j == len(dist)-1:
                    axs[j].set_title(r"Radius > %2.1f $\AA$" % (dist[j-1]*10), y=1.0, pad=-14)
                else:
                    axs[j].set_title(r"%2.1f <  Radius < %2.1f $\AA$" % (dist[j-1]*10, dist[j]*10), y=1.0, pad=-14)
        
                j+=1
                prevRes=np.array(resids)

            axs[j-1].set_xlabel("Residue index", fontsize=16)
            axs[j-1].xaxis.set_label_coords(-0.05, -.2)

            axs[j-1].set_ylabel("KLD similarity", fontsize=18)
            axs[j-1].yaxis.set_label_coords(-1.16, 2.3)

            plt.savefig(f"{frame}_KLD_backbone_radius.png", dpi=300)

            #plt.savefig(f"{frame}_KLD_sidechain_radius.png", dpi=300)
            #plt.close()
            #plt.show()

    if arg.simclose == True:    
        """
        Similaritites module
        """
        import mdtraj as md
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt

        frames=['f2000', 'f2700', 'f3400', 'f4100', 'f4800', 'f5500', 'f6200', 'f6900', 'f7600', 'f8300', 'f9000']
        
        fig, axs=plt.subplots(6, 2)
        fig.subplots_adjust(hspace = 0.2, wspace=0.1)
        fig.set_figheight(20)
        fig.set_figwidth(20)
        axs = axs.ravel()

        prevRes = [99999]
        j=0
        for count, frame in enumerate(frames):
            # Load PDB
            pdb = md.load_pdb(f"{frame}-1stFrame.pdb")
            chrome = pdb.topology.select('resname GYC')

            # Load similatities results
            simList=np.load(f"../results_sidechain/{frame}_KLD_sidechain.npy")
            simListBackbone=np.load(f"../results_backbone/{frame}_KLD_backbone.npy")

            #simList=simList-simListBackbone

            simRes=np.linspace(0, len(simList)-1, len(simList))
            
            dist = [0.4]

            resids, resnames = compute_neighbors(pdb,chrome,dist[0], False)
            resids.sort()

            currentRes=np.array(resids)
            layer=np.setdiff1d(currentRes, prevRes)

            layerRes=[]
            layerTest=[]
            for i in range(len(simList)):
                if simRes[i] in layer:
                    layerTest.append(int(simRes[i]+1))
                    layerRes.append(simList[i])

            x = np.arange(len(layerTest))
            wid=len(x)/70
            axs[j].bar(x, layerRes, width=wid)
            axs[j].set_xticks(x, layerTest, fontsize=9)
            axs[j].set_title(frame)
            
            j = j + 1

        plt.savefig(f"{frame}_KLD_backbone_radius.png", dpi=300)

        #plt.savefig(f"{frame}_KLD_sidechain_radius.png", dpi=300)
        #plt.close()
        #plt.show()


    if arg.torsion == True:

        #sys.path.insert(1, '/Users/rafael/theochem/projects/codes/mdtraj/mdtraj/geometry') 
        import hbond as hb
        import mdtraj as md 
        import numpy as np
        import matplotlib


        for frame in ['2000', '2700', '3400', '4100', '4800', '5500', '6200', '6900', '7600', '8300', '9000']:

            Itorsion=np.load(f"i_torsion-frame_{frame}.npy")
            Ptorsion=np.load(f"p_torsion-frame_{frame}.npy")

            print(frame)
            
            Imax=np.nanmax(Itorsion)
            Imin=np.nanmin(Itorsion)
            I=[]
            P=[]
            for  count, t in enumerate(Itorsion):
                if t == Imax:
                    print("Itorsion max", count, t)
                    I.append(Itorsion[count])
                    P.append(Ptorsion[count])
                elif t == Imin:
                    print("Itorsion min", count, t)
                    I.append(Itorsion[count])
                    P.append(Ptorsion[count])

            Pmax=np.nanmax(Ptorsion)
            Pmin=np.nanmin(Ptorsion)
            for  count, t in enumerate(Ptorsion):
                if t == Pmax:
                    print("Ptorsion max", count, t)
                    I.append(Itorsion[count])
                    P.append(Ptorsion[count])
                elif t == Pmin:
                    print("Ptorsion min", count, t)
                    I.append(Itorsion[count])
                    P.append(Ptorsion[count])


            PImax=np.nanmax(abs(Itorsion)+abs(Ptorsion))
            for  count, t in enumerate(abs(Itorsion)+abs(Ptorsion)):
                if t == PImax:
                    print("PItorsion max", count, t)
                    print("Itorsion", Itorsion[count])
                    print("Ptorsion", Ptorsion[count])
                    I.append(Itorsion[count])
                    P.append(Ptorsion[count])
        #print(PImax)
        #plt.plot(abs(Itorsion)+abs(Ptorsion))
        #plt.show()

            t=np.linspace(0, len(Itorsion)-1, len(Itorsion))*0.5
            alphas=np.linspace(1, 1, len(Itorsion))
            size=np.linspace(50, 50, len(Itorsion))


            # PLOT PROPAGATION
            plt.scatter(Itorsion,Ptorsion, c=t, cmap=cmc.hawaii, s=size, alpha=alphas, linewidth=0.1)
            # PLOT THE EXTREMES
            plt.scatter(I,P, c="red", s=80, marker='D')
            
            plt.ylabel('P-torsion')
            plt.xlabel('I-torsion')
            plt.xlim(-80,80)
            plt.ylim(-80,80) 
            plt.title('P-I-Torsion')
            cbar=plt.colorbar(values=t)
            cbar.set_label('Time (fs)')
            plt.savefig(f'i-p-torsion-2d-f{frame}.png', dpi=300)
            plt.close()
            #plt.show()

    if arg.dipole:
        import matplotlib
        import numpy as np
        import matplotlib.pyplot as plt

        # GET STEP (fs) AND ENERGY (a.u.)
        st, step, energy, dipole = get_tc_md_results(arg.dipole)
        
        s1=(energy[:,1]-energy[:,0])*27.211385
        d1=dipole[:,0]

        time=np.linspace(0,len(s1)-1,len(s1))/2


        plt.bar(s1,d1)

        #plt.plot(time,d1)
        plt.ylabel('Osc.strength (a.u.)')
        plt.xlabel('Time (fs)')
        plt.title('S0->S1 Oscillator strength')
        #plt.show()
        
        out=arg.dipole
        name=out.replace(".out", "-dipole.png")
        #plt.savefig(name, format='png', dpi=300)
        plt.show()

    if arg.spcplot == True:
        import hbond as hb
        import mdtraj as md 
        import numpy as np
        import matplotlib

        Emd=np.load("s0s1gap.npy")
        Emd=np.array(Emd).reshape(-1,2)

        Esp=[]
        with open("energies5.dat", 'r') as f_in:
            for line in f_in.readlines():
                T, E = line.split()
                Esp.append(float(E))
        
        time=np.linspace(0,len(Esp)-1, len(Esp))*2

        D=[]
        for i in range(len(Esp)):
            D.append(Esp[i]-Emd[i,0])
            #print(time[i], Esp[i])


        fig, ax = plt.subplots(1, 1)
        fig.set_figheight(5)
        fig.set_figwidth(22)

        #plt.plot(time,Esp, marker='.')
        #plt.plot((Emd[:,1]-1)/2,Emd[:,0], marker='.', color='r')
        plt.plot(time,D, color='r')
        plt.ylabel(r'$\Delta$E Protein/gas-phase (eV)  ')
        plt.xlabel('Time (fs)')
        #plt.title('S0->S1 Oscillator strength')
        plt.savefig("proteinVSgasphase.png", format='png', dpi=300)
        plt.show()


    if arg.violin == True:
        from matplotlib.ticker import MultipleLocator
        import matplotlib.pyplot as plt
        import numpy as np
    
        clr = 'r',
        edgeclr = 'darkred'

        # Sampled torsion
        X="P"
        # Other axis
        Y="P"

        # Conversions
        au_to_ps = 2.418884254E-5
        au_to_eV = 27.2114

        windows = np.arange(-100, 100+1, 10)

        if X == "I":
            windows = ['I' + str(w) for w in windows]
        else:
            windows = ['P' + str(w) for w in windows]

        for w in windows:
            Idihedrals = np.load('i_torsion_{}.npy'.format(w))
            Pdihedrals = np.load('p_torsion_{}.npy'.format(w))

            Idihedrals=np.nan_to_num(Idihedrals)
            Pdihedrals=np.nan_to_num(Pdihedrals)

            
            #if w == 'P-100':
            #    for i in range(len(Pdihedrals)):
            #        print(Pdihedrals[i])

            plt.figure(0)
            plt.axhline([0.0], ls='--', color='gray', alpha=0.8, zorder=0)

            xpos = int(w[1:])

            if Y == "I":
                parts = plt.violinplot(Idihedrals,[xpos],showmedians=True,showextrema=False,widths=7.0)
            else:
                parts = plt.violinplot(Pdihedrals,[xpos],showmedians=True,showextrema=False,widths=7.0)
            #for partname in ('cbars','cmins','cmaxes','cmedians'):
            
            for partname in ['cmedians']:
                vp = parts[partname]
                vp.set_edgecolor('black')
                vp.set_linewidth(1.5)
            for pc in parts['bodies']:
                pc.set_facecolor(clr)
                pc.set_alpha(0.8)
                pc.set_linewidth(1.5)
                #pc.set_edgecolor(edgeclr)
                pc.set_edgecolor('k')

        if X == "I":
            plt.xlabel(r'$\phi_I$ window (degrees)', fontsize=16)
        else:
            plt.xlabel(r'$\phi_P$ window (degrees)', fontsize=16)
        
        if Y == "I":
            plt.ylabel(r'$\phi_I$ samples (degrees)', fontsize=16)
        else:
            plt.ylabel(r'$\phi_P$ samples (degrees)', fontsize=16)
        
        plt.xticks(np.arange(-120, 120+1, 20), fontsize=14)
        plt.yticks(np.arange(-180, 180+1, 20), fontsize=14)
        plt.xlim([-110, 110])
        plt.ylim([-110, 110])
        #plt.axes().xaxis.set_minor_locator(MultipleLocator(10))
        #plt.axes().yaxis.set_minor_locator(MultipleLocator(10))
        plt.tight_layout()
        
        plt.savefig(f'violin_{X}_{Y}.png', dpi=400)
        
        #plt.show()

    if arg.violin2d == True:
        from matplotlib.ticker import MultipleLocator
        import matplotlib.pyplot as plt
        import numpy as np
    
        clr = 'r',
        edgeclr = 'darkred'

        # Sampled torsion
        X="P"
        # Other axis
        Y="P"

        # Conversions
        au_to_ps = 2.418884254E-5
        au_to_eV = 27.2114

        windows = np.arange(-100, 100+1, 10)

        #if X == "I":
        #    windows = ['I' + str(w) for w in windows]
        #else:
        #    windows = ['P' + str(w) for w in windows]

        for w in windows:
            Idihedrals = np.load('i_torsion_w{}.npy'.format(w))
            Pdihedrals = np.load('p_torsion_w{}.npy'.format(w))

            Idihedrals=np.nan_to_num(Idihedrals)
            Pdihedrals=np.nan_to_num(Pdihedrals)

            
            #if w == 'P-100':
            #    for i in range(len(Pdihedrals)):
            #        print(Pdihedrals[i])

            plt.figure(0)
            plt.axhline([0.0], ls='--', color='gray', alpha=0.8, zorder=0)

            xpos = int(w)

            if Y == "I":
                parts = plt.violinplot(Idihedrals,[xpos],showmedians=True,showextrema=False,widths=7.0)
            else:
                parts = plt.violinplot(Pdihedrals,[xpos],showmedians=True,showextrema=False,widths=7.0)
            #for partname in ('cbars','cmins','cmaxes','cmedians'):
            
            for partname in ['cmedians']:
                vp = parts[partname]
                vp.set_edgecolor('black')
                vp.set_linewidth(1.5)
            for pc in parts['bodies']:
                pc.set_facecolor(clr)
                pc.set_alpha(0.8)
                pc.set_linewidth(1.5)
                #pc.set_edgecolor(edgeclr)
                pc.set_edgecolor('k')

        #if X == "I":
        #    plt.xlabel('$\phi_I$ window (degrees)', fontsize=16)
        #else:
        plt.xlabel(r'$\phi_{IP}$ window (degrees)', fontsize=16)
        
        if Y == "I":
            plt.ylabel(r'$\phi_I$ samples (degrees)', fontsize=16)
        else:
            plt.ylabel(r'$\phi_P$ samples (degrees)', fontsize=16)
        
        plt.xticks(np.arange(-110, 110+1, 20), fontsize=14)
        plt.yticks(np.arange(-110, 110+1, 20), fontsize=14)
        plt.xlim([-110, 110])
        plt.ylim([-110, 110])
        #plt.axes().xaxis.set_minor_locator(MultipleLocator(10))
        #plt.axes().yaxis.set_minor_locator(MultipleLocator(10))
        plt.tight_layout()
        
        plt.savefig('violin.png', dpi=400)
        
        plt.show()

    if arg.efield == True:
        sys.path.insert(1, '/Users/rafael/theochem/projects/codes/Efield') 
        import geom
        import fileio

        # --> Gather mutated PDB files <-- #
        min_pdbs = []
        prmtops = []
        pdb_prmtops = {}

        min_dir = '/Users/rafael/theochem/projects/dronpa2/efield/pdbs'
        prmtop_dir = '/Users/rafael/theochem/projects/dronpa2/efield/prmtops'
        min_pdbs, prmtops, pdb_prmtops = fileio.gather_pdbs(min_dir, min_pdbs, prmtops, pdb_prmtops)
        print('Number of pdb files: {}'.format(len(min_pdbs)))
        print('Number of prmtop files: {}'.format(len(pdb_prmtops)))

        
        # --> Gather PDB data for WT GFP and all mutants <-- #
        print('\n# --> PDB file processing <-- #')
        min_pdb_files = None
        prmtop_files = None
        pdb_df_list = None

        print(min_pdbs)
        # Gather min and prmtop files
        min_pdb_files = ['{}/{}'.format(min_dir, min_pdb) for min_pdb in min_pdbs]
        #min_pdb_files.insert(0, wt_min)
        prmtop_files = ['{}/{}'.format(prmtop_dir, prmtop) for prmtop in prmtops] 
        #prmtop_files.insert(0, wt_prmtop)
        assert(len(min_pdb_files) == len(prmtop_files))

        print(min_pdb_files)

        # Store all pdb data as a list of dataframes
        pdb_df_list = fileio.process_pdb_data(min_pdb_files, nproc=3, taskset=None)

        # Pickle intermediate data
        fileio.pickle_data(min_pdb_files, 'min_pdb_files.pkl')
        fileio.pickle_data(prmtop_files, 'prmtop_files.pkl')
        fileio.pickle_data(pdb_df_list, 'pdb_df_list.pkl')

        
        # --> Partition chromophore by moiety <-- #
        chr_seq = 'GYC'
        
        chr_atomnames = ['N','H','C1','H1','C3','H3','H5','SH','HS','C','O','C2','H2','H4', # caps
                        'CA','NA','CB','OA','NB','CC', # I-ring
                        'CD','HD', #bridge
                        'CE','CF','FH','CG','HG','CH','FF','CI','FI','CJ','OK'] #P-ring
        print('Number of chromophore atoms: ', len(chr_atomnames))

        # Chromophore ring moieties - heavy atom names
        print('\n# --> Chromophore index handling <-- #')
        Iring_atoms = ['NA','CA','CB','NB','CC', 'OA']
        Pring_atoms = ['CE','CF','FH','CG','CH','FF','CI','FI','CJ','OK']
        bridge_atoms = ['CD']

        # Chromophore ring moieties - indices
        Iring_idxs = [chr_atomnames.index(name) for name in Iring_atoms]
        Pring_idxs = [chr_atomnames.index(name) for name in Pring_atoms]
        bridge_idxs = [chr_atomnames.index(name) for name in bridge_atoms]

        print('Iring idxs: {}'.format(Iring_idxs))
        print('Pring idxs: {}'.format(Pring_idxs))
        print('bridge idxs: {}'.format(bridge_idxs))

        
        # --> Parition system into chromophore and non-chromophore regions <-- #
        print('\n# --> Partitioning: chromophore data <-- #')
        chr_Xs = None
        chr_Ys = None
        chr_Zs = None
        chr_XYZs = None
        chr_Ms = None
        chr_Qs = None

        chr_Xs, chr_Ys, chr_Zs, chr_XYZs, chr_Ms, chr_Qs = geom.obtain_chr_data(pdb_df_list, prmtop_files, chr_seq, chr_atomnames)

        
        # Pickle intermediate data
        fileio.pickle_data(chr_Xs, 'chr_Xs.pkl')
        fileio.pickle_data(chr_Ys, 'chr_Ys.pkl')
        fileio.pickle_data(chr_Zs, 'chr_Zs.pkl')
        fileio.pickle_data(chr_XYZs, 'chr_XYZs.pkl')
        fileio.pickle_data(chr_Ms, 'chr_Ms.pkl')
        fileio.pickle_data(chr_Qs, 'chr_Qs.pkl')

        
        # --> Parition system into chromophore and non-chromophore regions <-- #
        print('\n# --> Partitioning: non-chromophore data <-- #')
        nonchr_residues = None
        nonchr_XYZs = None
        nonchr_Ms = None
        nonchr_Qs = None

        nonchr_residues, nonchr_XYZs, nonchr_Ms, nonchr_Qs = geom.obtain_nonchr_data(pdb_df_list, prmtop_files, chr_seq)

        # Pickle intermediate data
        fileio.pickle_data(nonchr_residues, 'nonchr_residues.pkl')
        fileio.pickle_data(nonchr_XYZs, 'nonchr_XYZs.pkl')
        fileio.pickle_data(nonchr_Ms, 'nonchr_Ms.pkl')
        fileio.pickle_data(nonchr_Qs, 'nonchr_Qs.pkl')

        assert(len(chr_XYZs) == len(nonchr_XYZs))

        
        # --> Obtain Efield probe vectors on chromophore atoms <-- #
        print('\n# --> Calculating probe vectors for Iring, Pring, bridge atoms <-- #')
        Iring_CO_idxs = [chr_atomnames.index('CJ'), chr_atomnames.index('OK')]
        Pring_CO_idxs = [chr_atomnames.index('CB'), chr_atomnames.index('OA')]
        bridge_CH_idxs = [chr_atomnames.index('CD'), chr_atomnames.index('HD')]

        print(chr_XYZs[0][31])
        Iring_CO_midpts = geom.obtain_probe_point(chr_XYZs, Iring_CO_idxs)
        Pring_CO_midpts = geom.obtain_probe_point(chr_XYZs, Pring_CO_idxs)
        bridge_CH_midpts = geom.obtain_probe_point(chr_XYZs, bridge_CH_idxs)

        Iring_CO_vectors = geom.obtain_probe_vector(chr_XYZs, Iring_CO_idxs)
        Pring_CO_vectors = geom.obtain_probe_vector(chr_XYZs, Pring_CO_idxs)
        bridge_CH_vectors = geom.obtain_probe_vector(chr_XYZs, bridge_CH_idxs)


        # --> Calculate protein electric field acting along probe vectors <-- #
        print('\n# --> Calculating Efield along probe vectors <-- #')
        df_Iring_CO_Efield = None
        df_Pring_CO_Efield = None
        df_bridge_CH_Efield = None

        Iring_CO_Efield = geom.calculate_Efield(Iring_CO_midpts, Iring_CO_vectors, chr_XYZs, chr_Qs, nonchr_XYZs, nonchr_Qs)
        Pring_CO_Efield = geom.calculate_Efield(Pring_CO_midpts, Pring_CO_vectors, chr_XYZs, chr_Qs, nonchr_XYZs, nonchr_Qs)
        bridge_CH_Efield = geom.calculate_Efield(bridge_CH_midpts, bridge_CH_vectors, chr_XYZs, chr_Qs, nonchr_XYZs, nonchr_Qs)
        
        # Store resulting data in dataframe
        print('Generating dataframes...')
        df_Iring_CO_Efield = pd.DataFrame(Iring_CO_Efield, columns=['Efield_Iring_CO (MV/cm)'])
        df_Pring_CO_Efield = pd.DataFrame(Pring_CO_Efield, columns=['Efield_Pring_CO (MV/cm)'])
        df_bridge_CH_Efield = pd.DataFrame(bridge_CH_Efield, columns=['Efield_bridge_CH (MV/cm)'])

        # Pickle intermediate data
        fileio.pickle_data(df_Iring_CO_Efield, 'df_Iring_CO_Efield.pkl')
        fileio.pickle_data(df_Pring_CO_Efield, 'df_Pring_CO_Efield.pkl')
        fileio.pickle_data(df_bridge_CH_Efield, 'df_bridge_CH_Efield.pkl')
        
        print(df_Iring_CO_Efield)
        print(df_Pring_CO_Efield)
        print(df_bridge_CH_Efield)
        

    if arg.velscale == True:
        import matplotlib.pyplot as plt
        import numpy as np
        import mdtraj as md
        import math

        # LOAD VELOCITY AND PRMTOP FILES
        traj=md.load('vel.dcd', top='../sphere.prmtop')

        lastFrame=len(traj)-1
        secLastFrame=len(traj)-2

        # HYDROGEN AND FLUORIDE MASSES
        mF=18.9984031627
        mH=1.00782503223

        #SCALE
        scale=math.sqrt(mF/mH)

        #LOOP OVER THE DIFFERENT F/H
        Hs=[950,954,956]
        for h in Hs:

            #dTdF=(traj.xyz[lastFrame][h]-traj.xyz[secLastFrame][h])*2*mF
            #vi_fdf_H=traj.xyz[secLastFrame][h]+(dTdF/(2*mH))
            #traj.xyz[lastFrame][h]=vi_fdf_H
            traj.xyz[lastFrame][h]=traj.xyz[lastFrame][h]*scale


        Traj=traj[lastFrame]
        Traj.save_amberrst7('ScaledVel.rst7')
    
    if arg.meci3 == True:
        import mdtraj as md 
        import socket, glob
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        from matplotlib.lines import Line2D

         # ON MACMINI    
        if socket.gethostname() == "rcc-mac.kemi.kth.se":
            sys.path.insert(1, '/Users/rafael/theochem/projects/codes/tcutil/code/geom_param') 
        # ON BERZELIUS
        else:
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
        import geom_param as gp

        # Chromophore indices
        chrome=[924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]

        # TORSION INDEXES
        i_pair=[22,24]
        i_triple=[21,20,18]
        p_pair=[22,21]
        p_triple=[24,27,25]

        # PYRAMIDALIZATION INDEXES  
        pyr_idx= [22,23,24,21]

        # SETUP FIG
        fig, ax = plt.subplots()
        
        mecifiles=sorted(glob.iglob('s1min_From_meci*.dcd'))
        # COLORS
        color = cm.Paired(np.linspace(0, 1, len(mecifiles)))
        
        for mecifile, c in zip(mecifiles, color):
            # SETUP FIG
            #fig, ax = plt.subplots()

            # MECI FILES
            meciprmtop=mecifile.replace(".dcd", ".prmtop")

            # LOAD MECI TRAJECTORY
            MECItop = md.load_prmtop(meciprmtop)
            MECItraj = md.load_dcd(mecifile, top = MECItop)


            # INITIAL GEOMETRY
            Iteta_i = gp.compute_torsion5(MECItraj.xyz[0,chrome,:],i_pair,i_triple)
            Iteta_p = gp.compute_torsion5(MECItraj.xyz[0,chrome,:],p_pair,p_triple)
            Iteta_pyr = gp.compute_pyramidalization(MECItraj.xyz[0,chrome,:],pyr_idx[0],pyr_idx[1],pyr_idx[2],pyr_idx[3]) 
            scatter = ax.scatter(Iteta_i,Iteta_p, s=100, c=Iteta_pyr, cmap='coolwarm', alpha=1, vmin=-40, vmax=40,  edgecolors='black', linewidths=1)

            # FINAL MECI
            N=len(MECItraj)-1
            MECIteta_i   = gp.compute_torsion5(MECItraj.xyz[N,chrome,:],i_pair,i_triple)
            MECIteta_p   = gp.compute_torsion5(MECItraj.xyz[N,chrome,:],p_pair,p_triple)
            MECIteta_pyr = gp.compute_pyramidalization(MECItraj.xyz[N,chrome,:],pyr_idx[0],pyr_idx[1],pyr_idx[2],pyr_idx[3])
            scatter = ax.scatter(MECIteta_i,MECIteta_p, s=100, c=MECIteta_pyr, cmap='coolwarm', alpha=1, vmin=-40, vmax=40)
            

            # CONNECTING LINE INIT -> S1
            ax.plot([Iteta_i, MECIteta_i], [Iteta_p, MECIteta_p], ls="--", lw="0.8", c=".01", alpha=0.1)

        
            #cbar=plt.colorbar(scatter)
            #cbar.set_label(r'$\theta_{pyr}$ (degrees)',fontsize=14)
            #plt.title(name)
            plt.xlabel(r"$\phi_I$ (degrees)",fontsize=14)
            plt.ylabel(r"$\phi_P$ (degrees)",fontsize=14)

            #plt.savefig(f's1-meci-{name}.png', dpi=300, format='png')
            #plt.show(block = True)
            #plt.close()

        cbar=plt.colorbar(scatter)
        cbar.set_label(r'$\theta_{pyr}$ (degrees)',fontsize=14)
        #plt.savefig(f's1-meci-ALL.png', dpi=300, format='png')
        plt.show(block = True)

    if arg.usdist:
        #sys.path.insert(1, '/Users/rafael/theochem/projects/codes/mdtraj/mdtraj/geometry') 
        import hbond as hb
        import mdtraj as md 
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt

        sample=['-100', '-90','-80','-70','-60','-50','-40','-30','-20','-10','0','10','20','30','40','50','60','70','80','90','100']

        colors=['black', 'silver', 'red',  'salmon', 'brown', 'orange', 'gold','yellow', 'olive', 'green', 'lime', 'teal', 'aqua', 'blue', 'navy', 'violet', 'lavender', 'magenta', 'pink', 'purple', 'black']

        torsion=arg.usdist
        fig, ax = plt.subplots()
        for count, frame in enumerate(sample):

            Itorsion=np.load(f"i_torsion_{torsion}{frame}.npy")
            Ptorsion=np.load(f"p_torsion_{torsion}{frame}.npy")

            # PLOT PROPAGATION
            #plt.scatter(Itorsion,Ptorsion, c=colors[count], label=sample[count])
            scatter = ax.scatter(Itorsion,Ptorsion, c=colors[count], label=sample[count])

        


        fig.set_figheight(7)
        fig.set_figwidth(9)

        #plt.legend(handles=colors, labels=sample, loc='upper left', frameon=False)
        plt.ylabel('P-torsion (deg)')
        plt.xlabel('I-torsion (deg)')
        plt.xlim(-110,110)
        plt.ylim(-110,110)
        plt.title(f'{torsion}-torsion sampling')
        plt.legend(loc=1, ncol=2, title='Window  (deg)')

        plt.savefig(f'US-scatter-{torsion}.png', dpi=300)
        plt.show()

    if arg.com == True:
        import MDAnalysis as mda
        from MDAnalysis.analysis import rdf, distances
        from matplotlib.ticker import MultipleLocator
        from matplotlib import pyplot as plt
        import numpy as np
        import seaborn as sns
        import mdtraj as md 
        import warnings, socket
        import cmcrameri.cm as cmc
        from matplotlib.cm import ScalarMappable


        #sys.path.insert(1, '/home/rcouto/theochem/progs/tcutil/code/geom_param')
        
        sys.path.insert(1, '/Users/rafael/theochem/projects/codes/tcutil/code/geom_param') 
        import geom_param as gp
        
        # suppress some MDAnalysis warnings about PSF files
        warnings.filterwarnings('ignore')

        #################
        #dcd='all_opt.dcd'
        #prmtop='sphere.prmtop'
        #dcd='s1dyn_Htfdronpra2_f27.dcd'
        #prmtop='s1dyn_Htfdronpra2_f27.prmtop'
        #dcd='s1dyn_f27.dcd'
        #prmtop='s1dyn_f27.prmtop'
        #dcd='meci-f27-I-70.dcd'
        #prmtop='meci-f27-I-70.prmtop'
        dcd='coors.dcd'
        prmtop='sphere.prmtop'
        #################

        #u=mda.Universe('sphere.prmtop', dcd)
        u=mda.Universe(prmtop, dcd)


        

        pring=u.atoms[[948, 949, 951, 953, 955, 957]]
        his190=u.atoms[[2993,2994,2996,2998,3000]]
        trp86=u.atoms[[1367,1368,1370,1372,1374,1376]]

        #ok=u.atoms[[958]]
        #ser=u.atoms[[2186]]
        #h2o=u.atoms[[6990]]

        """
        residue_list=[2988, # HIS_190
               878,  # THR_58
               3023, # ILE_192
               956,  # FI
               950,  # FH
               947,  # HD (Bridge)
               946,  # CD (Bridge)
               3284, # GLU_208
               2205, # GLU_141
               [989,943],  # ARG_63_N1->GYC-OA
               [992,943],  # ARG_63_N2->GYC-OA
               [1413,943], # ARG_88_N2->GYC-OA
               [1410,943],  # ARG_88_N1->GYC-OA
               ]
        """
        residue_list=[
               [989,943],  # ARG_63_N1->GYC-OA
               [992,943],  # ARG_63_N2->GYC-OA
               [1413,943], # ARG_88_N2->GYC-OA
               [1410,943],  # ARG_88_N1->GYC-OA
               ]

        #topology = md.load_prmtop(prmtop)
        #traj = md.load_dcd(dcd, top = topology)
        # Chromophore indices
        chrome=[924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]

        traj=u.atoms[[chrome]]

        # Radian to degrees
        r2d=57.2958
        p_torsion=[]
        i_torsion=[]
        # Related atoms
        i_pair=[22,24]
        i_triple=[21,20,18]
        p_pair=[22,21]
        p_triple=[24,27,25]
        DCOM=[]
        serHB=[]
        h2oHB=[]

        residues_dist=np.zeros([len(residue_list), len(u.trajectory)])
        
        for i, _ in enumerate(u.trajectory):

            pring_com = pring.center_of_mass(compound='group')
            his190_com = his190.center_of_mass(compound='group')
            trp86_com = trp86.center_of_mass(compound='group')
    
            #print(gyc_com, hip_com)
            dist_com = distances.distance_array(pring_com, his190_com, box=u.dimensions)
            DCOM.append(float(dist_com[0]))

            dist_h2o=distances.distance_array(ok, h2o, box=u.dimensions)
            h2oHB.append(float(dist_h2o[0]))
            dist_ser=distances.distance_array(ok, ser, box=u.dimensions)
            serHB.append(float(dist_ser[0]))

            for j, res in enumerate(residue_list):
                atom1 = u.atoms[[res[0]]]
                atom2 = u.atoms[[res[1]]]
                residues_dist[j][i] = distances.distance_array(atom1, atom2, box=u.dimensions)
                
            # Compute the I- and P-torsion angles          
            # I-torsion
            #teta = gp.compute_torsion5(traj.xyz[count,chrome,:],i_pair,i_triple)
            #print(traj.positions)
            teta = gp.compute_torsion5(traj.positions[0],i_pair,i_triple)
            i_torsion.append(teta)
            
            # P-torsion
            #teta = gp.compute_torsion5(traj.xyz[count,chrome,:],p_pair,p_triple)
            teta = gp.compute_torsion5(traj.positions[0],p_pair,p_triple)
            p_torsion.append(teta)

        #fig, ax = plt.subplots()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        X=np.linspace(0, len(u.trajectory)-1, len(u.trajectory))

        #ax=sns.scatterplot(x=i_torsion, y=DCOM, hue=X, markers=X, alpha=0.5, palette=cmc.hawaii, sizes=X, s=100, legend=False)
        #ax=sns.kdeplot(x=i_torsion, y=DCOM, fill=True, alpha=0.5)
        ax1=sns.jointplot(x=i_torsion, y=DCOM, kind='kde', color='purple', alpha=0.5)
        ax1=sns.scatterplot(x=i_torsion, y=DCOM, hue=X, markers=X, palette=cmc.hawaii, sizes=X, s=100, legend=False, linewidth=0, alpha=0.8)

        ax1.set(xlabel=r"$\phi_P$ (deg)", ylabel=r"$R_{P-ring - HIP}$ ($\AA$)")

        #Colormap for comparison
        cmap = plt.get_cmap(cmc.hawaii)
        norm = plt.Normalize(X[0],X[-1])
        sm =  ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1)
        plt.savefig('cm.png')

        ax2=sns.jointplot(x=i_torsion, y=serHB, kind='kde', color='purple', alpha=0.5)
        ax2=sns.scatterplot(x=i_torsion, y=serHB, hue=X, markers=X, palette=cmc.hawaii, sizes=X, s=100, legend=False, linewidth=0, alpha=0.8)
        
        #Colormap for comparison
        #cmap = plt.get_cmap(cmc.hawaii)
        #norm = plt.Normalize(X[0],X[-1])
        #sm =  ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2)
        plt.savefig('serHB.png')
        
        ax3=sns.jointplot(x=i_torsion, y=h2oHB, kind='kde', color='purple', alpha=0.5)
        ax3=sns.scatterplot(x=i_torsion, y=h2oHB, hue=X, markers=X, palette=cmc.hawaii, sizes=X, s=100, legend=False, linewidth=0, alpha=0.8)
        
        #Colormap for comparison
        #cmap = plt.get_cmap(cmc.hawaii)
        #norm = plt.Normalize(X[0],X[-1])
        #sm =  ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax3)
        plt.savefig('h2oHB.png')

        #plt.show()

        fig, ax = plt.subplots()
        plt.tight_layout()
        residues_names = ['ARG63-N1_GYC-OA', 'ARG63-N2_GYC-OA', 'ARG88-N2_GYC-OA', 'ARG88-N1_GYC-OA']

        for i, name in enumerate(residues_names):

            ax=sns.jointplot(x=i_torsion, y=residues_dist[i][:], kind='kde', color='purple', alpha=0.5)
            ax=sns.scatterplot(x=i_torsion, y=residues_dist[i][:], hue=X, markers=X, palette=cmc.hawaii, sizes=X, s=100, legend=False, linewidth=0, alpha=0.8)
            ax.set(xlabel=r"$\phi_I$ (deg)", ylabel=fr"Dist. {name} (\AA)")

            #Colormap for comparison
            cmap = plt.get_cmap(cmc.hawaii)
            norm = plt.Normalize(X[0],X[-1])
            sm =  ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            plt.savefig(f'{name}_I.png', bbox_inches="tight")
            plt.close()

        for i, name in enumerate(residues_names):

            ax=sns.jointplot(x=p_torsion, y=residues_dist[i][:], kind='kde', color='purple', alpha=0.5)
            ax=sns.scatterplot(x=p_torsion, y=residues_dist[i][:], hue=X, markers=X, palette=cmc.hawaii, sizes=X, s=100, legend=False, linewidth=0, alpha=0.8)
            ax.set(xlabel=r"$\phi_P$ (deg)", ylabel=fr"Dist. {name} (\AA)")

            #Colormap for comparison
            cmap = plt.get_cmap(cmc.hawaii)
            norm = plt.Normalize(X[0],X[-1])
            sm =  ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            plt.savefig(f'{name}_P.png', bbox_inches="tight")
            plt.close()


    if arg.distances == True:
        import MDAnalysis as mda
        from MDAnalysis.analysis import rdf, distances
        from matplotlib.ticker import MultipleLocator
        from matplotlib import pyplot as plt
        import numpy as np
        import seaborn as sns
        import warnings, sys, socket
        import cmcrameri.cm as cmc
        from matplotlib.cm import ScalarMappable

        if socket.gethostname() == "nhlist-desktop":
            sys.path.insert(1, '/home/rcouto/theochem/progs/tcutil/code/geom_param')
        elif socket.gethostname() == "berzelius2.nsc.liu.se":
            sys.path.insert(1, '/proj/berzelius-2023-33/users/x_rafca/progs/tcutil/code/geom_param')
        elif socket.gethostname() == "amaze":
            sys.path.insert(1, '/data/users/rcc/codes/tcutil/code/geom_param')
        else:
            sys.path.insert(1, '/Users/rafael/theochem/projects/codes/tcutil/code/geom_param') 
        import geom_param as gp
        
        # suppress some MDAnalysis warnings about PSF files
        warnings.filterwarnings('ignore')

        #################
        #dcd='dronpa2_US_I0.dcd'
        #prmtop='dronpa2_US_I0.prmtop'
        #################
        
        # LOAD TRAJECTORY
        if arg.dcd2:
            u=mda.Universe(arg.top, [arg.dcd,arg.dcd2])
        elif arg.dcdlist:
            dcds = [ file.rstrip('\n') for file in open(arg.dcdlist, 'r').readlines() ]
            u=mda.Universe(arg.top, dcds)
        else:
            u=mda.Universe(arg.top, arg.dcd)

        #u=mda.Universe('sphere.prmtop', dcd)
        #u=mda.Universe(prmtop, dcd)

        connections=read_table("index_dist_final.dat")

        # TF-Dronpa2
        pring=u.atoms[[948,949,953,957,955,951]]
        trp136=u.atoms[[2137,2138,2140,2142,2144,2146]]
        his190=u.atoms[[2993,2994,2996,2998,3000]]
        trp86=u.atoms[[1367,1368,1370,1372,1374,1376]]
        thr58_chg2=u.atoms[[882]]
        
        # Dronpa2 and TF-Dronpa2 - MUTATION THR58->GLY58
        #pring=u.atoms[[940,941,945,949,947,943]]
        #trp136=u.atoms[[2129,2130,2132,2134,2136,2138]]
        #his190=u.atoms[[2985,2986,2988,2990,2992]]
        #trp86=u.atoms[[1359,1360,1362,1364,1366,1368]]
        

        # Chromophore indices
        chrome=[924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]

        # Chromophore indices - Dronpa2 and TF-Dronpa2 - MUTATION THR58->GLY58
        #chrome=[917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953]

        traj=u.atoms[[chrome]]
        
        # PHE170
        phe170=u.select_atoms("resid 170")
        phe170_pair=[3,1]
        phe170_triple=[6,15,7]
        phe170_dihedral=[]

        # ANGLE BETWEEN HIS190_COM GYC_PRING_COM THR58_CG2
        his_gyc_thr=[]

        # Radian to degrees
        r2d=57.2958
        p_torsion=[]
        i_torsion=[]
        pyra=[]
        # Related atoms
        i_pair=[22,24]
        i_triple=[21,20,18]
        p_pair=[22,21]
        p_triple=[24,27,25]
        DCOM=[]

        # FOR MINIMUM DISTANCE
        gyc61=u.select_atoms("resid 61")
        glu211=u.select_atoms("resid 211")
        glu144=u.select_atoms("resid 144")

        phe170_out=open('GYC61_PHE170_min_dist.dat', 'w')
        glu211_out=open('GYC61_GLU211_min_dist.dat', 'w')
        glu144_out=open('GYC61_GLU144_min_dist.dat', 'w')

        phe170_dist=[]
        glu211_dist=[]
        glu144_dist=[]

        all_distances=np.zeros([len(connections), len(u.trajectory)])
        
        for i, _ in enumerate(u.trajectory):

            pring_com = pring.center_of_mass(compound='group')
            his190_com = his190.center_of_mass(compound='group')
            trp86_com = trp86.center_of_mass(compound='group')
            trp136_com = trp136.center_of_mass(compound='group')

            for j, connect in enumerate(connections):

                if connect[1].isdigit():
                    atom1 = u.atoms[[int(connect[1])]].positions
                else:
                    if connect[1] == 'his190_com':
                        atom1 = his190_com
                    elif connect[1] == 'pring_com':
                        atom1 = pring_com
                    elif connect[1] == 'trp86_com':
                        atom1 = trp86_com
                    elif connect[1] == 'trp136_com':
                        atom1 = trp86_com
                    else:
                        print(f"Center of mass {connect[1]} not available.")
                        sys.exit(1)

                if connect[2].isdigit():
                    atom2 = u.atoms[[int(connect[2])]].positions
                else:
                    if connect[2] == 'his190_com':
                        atom2 = his190_com
                    elif connect[2] == 'pring_com':
                        atom2 = pring_com
                    elif connect[2] == 'trp86_com':
                        atom2 = trp86_com
                    elif connect[2] == 'trp136_com':
                        atom2 = trp136_com
                    else:
                        print(f"Center of mass {connect[2]} not available.")
                        sys.exit(1)
 
                all_distances[j][i] = distances.distance_array(reference=atom1, configuration=atom2, box=u.dimensions)
            
            # Compute the I- and P-torsion angles          
            # I-torsion
            theta = gp.compute_torsion5(traj.positions[0],i_pair,i_triple)
            i_torsion.append(theta)
            # P-torsion
            theta = gp.compute_torsion5(traj.positions[0],p_pair,p_triple)
            p_torsion.append(theta)
            # Bridge piramidalization
            theta = gp.compute_pyramidalization(traj.positions[0],22,23,24,21)
            pyra.append(theta)

            # PHE170 ring rotation
            theta=gp.compute_torsion5(phe170.positions,phe170_pair,phe170_triple)
            phe170_dihedral.append(theta)

            # ANGLE BETWEEN HIS190_COM GYC_PRING_COM THR58_CG2
            theta = three_points_angle(his190_com, pring_com, thr58_chg2.positions[0])
            his_gyc_thr.append(theta)
        
            # MINIMUM DISTANCE BETWEEN GYC AND PHE/GLU
            d=mda.analysis.distances.distance_array(gyc61,phe170)
            min_dist=np.min(d)
            index=np.argwhere(d == min_dist)
            phe170_dist.append(min_dist)
            phe170_out.write('GYC61-{:s} PHE170-{:s}: {:>2.2f} \n'.format(gyc61[index[0][0]].name, phe170[index[0][1]].name, min_dist))

            d=mda.analysis.distances.distance_array(gyc61,glu211)
            min_dist=np.min(d)
            index=np.argwhere(d == min_dist)
            glu211_dist.append(min_dist)
            glu211_out.write('GYC61-{:s} GLU211-{:s}: {:>2.2f} \n'.format(gyc61[index[0][0]].name, glu211[index[0][1]].name, min_dist))

            d=mda.analysis.distances.distance_array(gyc61,glu144)
            min_dist=np.min(d)
            index=np.argwhere(d == min_dist)
            glu144_dist.append(min_dist)
            glu144_out.write('GYC61-{:s} GLU144-{:s}: {:>2.2f} \n'.format(gyc61[index[0][0]].name, glu144[index[0][1]].name, min_dist))


        phe170_out.close()
        glu211_out.close()
        glu144_out.close()

        np.save('GYC61_PHE170_min_dist.npy', phe170_dist)
        np.save('GYC61_GLU211_min_dist.npy', glu211_dist)
        np.save('GYC61_GLU144_min_dist.npy', glu144_dist)

        if arg.mmd:
            # MD ANALYSIS
            T=np.linspace(0,len(u.trajectory),len(u.trajectory))/100
        else:            
            T=np.linspace(0,len(u.trajectory)/2,len(u.trajectory))
        
        out = open("distances_summary.dat", 'w')
        out.write("# Connection                  min (A) max (A)  average (A) deviation\n")
        

        # PLOT INDIVIDUAL CONTRIBUTIONS
        for i in range(len(connections)):   
            fig, ax = plt.subplots(2,1)
            fig.set_figheight(10)
            fig.set_figwidth(15)

            avrg=np.mean(all_distances[i][:])
            ax[0].plot([0, T[-1]], [avrg, avrg], ls="-", lw="0.8", c="red", label="mean={:>2.1f}".format(avrg))
            
            ax[0].plot(T, all_distances[i][:], label=connections[i][0])
            ax[0].set_ylabel(r'Distance ($\AA$)')
            if arg.name:
                ax[0].set_title(f'{arg.name} - Distance {connections[i][0]}', fontsize=20)
            else:
                ax[0].set_title(f'Distance {connections[i][0]}', fontsize=20)
            ax[0].set_xticklabels([])
            ax[0].legend(loc='upper right', framealpha=0.5)

            ax[1].plot(T,i_torsion, label='I-torsion')
            ax[1].plot(T,p_torsion, label='P-torsion')
            ax[1].set_ylabel(r'$\phi$ (deg)')

            color = 'tab:green'
            ax2 = ax[1].twinx()
            ax2.plot(T, pyra, color=color, ls='--', lw=1, alpha=0.5, label='Pyr.')
            ax2.set_ylabel(r'Pyramidalization (deg)')
            ax2.set_ylim(-40,40)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.legend(loc='upper right', framealpha=0.5)

            if arg.mmd:
                # MD ANALYSIS
                ax[1].set_xlabel('Time (ns)')
            else:
                ax[1].set_xlabel('Time (fs)')

            ax[1].legend(loc='upper left', framealpha=0.5)
            
            ax[0].set_xlim(0,T[-1])
            ax[1].set_xlim(0,T[-1])
            ax[1].set_ylim(-80,80)

            plt.subplots_adjust(hspace=0)
            if arg.name:
                plt.savefig(f'{arg.name}_{connections[i][0]}.png', dpi=300)
            else:
                plt.savefig(f'{connections[i][0]}.png', dpi=300)
            plt.close()

            # WRITE SUMMARY VALUES TO FILE
            min=np.min(all_distances[i][:])
            max=np.max(all_distances[i][:])
            dev=np.std(all_distances[i][:])
            out.write('{:<12s}\t\t{:>2.2f}\t{:>2.2f}\t{:>2.2f}\t{:>2.2f}\n'.format(str(connections[i][0]),  min, max, avrg, dev))


            if connections[i][0] == 'PHE170_HZ-GYC_Pring_COM':
                fig, ax = plt.subplots(2,1)
                fig.set_figheight(10)
                fig.set_figwidth(15)

                ax[0].plot(T, all_distances[i][:], label=connections[i][0])

                avrg=np.mean(all_distances[i][:])
                ax[0].plot([0, T[-1]], [avrg, avrg], ls="-", lw="0.8", c="red", label="mean={:>2.1f}".format(avrg))

                ax[0].set_ylabel(r'Distance ($\AA$)')
                if arg.name:
                    ax[0].set_title(f'{arg.name} - Distance {connections[i][0]}', fontsize=20)
                else:
                    ax[0].set_title(f'Distance {connections[i][0]} + PHE170 ring dihedral', fontsize=20)
                ax[0].set_xticklabels([])
                ax[0].legend(loc='upper right', framealpha=0.5)

                min=np.min(all_distances[i][:])
                max=np.max(all_distances[i][:])
                dev=np.std(all_distances[i][:])
                out.write('{:<12s}\t\t{:>2.2f}\t{:>2.2f}\t{:>2.2f}\t{:>2.2f}\n'.format(str(connections[i][0]),  min, max, avrg, dev))


                ax[1].plot(T,phe170_dihedral, label='PHE170 ring dihedral')
                ax[1].set_ylabel(r'PHE170 $\phi$ (deg)')

                min=np.min(phe170_dihedral)
                max=np.max(phe170_dihedral)
                avrg=np.mean(phe170_dihedral)
                phe170label='PHE170_ring_dihedral'
                dev=np.std(all_distances[i][:])
                out.write('{:<12s}\t\t{:>2.2f}\t{:>2.2f}\t{:>2.2f}\t{:>2.2f}\n'.format(str(connections[i][0]),  min, max, avrg, dev))

                if arg.mmd:
                # MD ANALYSIS
                    ax[1].set_xlabel('Time (ns)')
                else:
                    ax[1].set_xlabel('Time (fs)')

                ax[1].legend(loc='upper left', framealpha=0.5)
                
                ax[0].set_xlim(0,T[-1])
                ax[1].set_xlim(0,T[-1])

                plt.subplots_adjust(hspace=0)
                if arg.name:
                    plt.savefig(f'{arg.name}_{connections[i][0]}_RING.png', dpi=300)
                else:
                    plt.savefig(f'{connections[i][0]}_RING.png', dpi=300)
                plt.close()
        out.close()
        
        # PLOT DISTANCE ONLY
        for i in range(len(connections)):
            fig, ax = plt.subplots()
            fig.set_figheight(5)
            fig.set_figwidth(10)

            ax.plot(T, all_distances[i][:], label=connections[i][0])

            avrg=np.mean(all_distances[i][:])
            ax.plot([0, T[-1]], [avrg, avrg], ls="-", lw="0.8", c="red", label="mean={:>2.1f}".format(avrg))

            ax.set_ylabel(r'Distance ($\AA$)')

            if arg.mmd:
                # MD ANALYSIS
                    ax.set_xlabel('Time (ns)')
            else:
                    ax.set_xlabel('Time (fs)')

            if arg.name:
                ax.set_title(f'{arg.name} - Distance {connections[i][0]}', fontsize=15)
            else:
                ax.set_title(f'Distance {connections[i][0]}', fontsize=15)
            ax.legend(loc='upper right', framealpha=0.5)
            ax.set_xlim(0,T[-1])

            if arg.name:
                plt.savefig(f'{arg.name}_{connections[i][0]}_ONLY.png', dpi=300)
            else:
                plt.savefig(f'{connections[i][0]}_ONLY.png', dpi=300)
            plt.close()
            np.save(f'{connections[i][0]}.npy', all_distances[i][:])


        # PLOT GYC61-PHE170 DISTANCE
        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(10)
        ax.plot(T, phe170_dist, label='GYC61-PHE170')
        avrg=np.mean(phe170_dist)
        ax.plot([0, T[-1]], [avrg, avrg], ls="-", lw="0.8", c="red", label="mean={:>2.1f}".format(avrg))
        ax.set_ylabel(r'Distance ($\AA$)')
        if arg.mmd:
            # MD ANALYSIS
                ax.set_xlabel('Time (ns)')
        else:
                ax.set_xlabel('Time (fs)')
        ax.set_title(f'Distance GYC61-PHE170', fontsize=15)
        ax.legend(loc='upper right', framealpha=0.5)
        ax.set_xlim(0,T[-1])
        plt.savefig(f'GYC61_PHE170_min_dist.png', dpi=300)
        plt.close()

        # PLOT GYC61-GLU211 DISTANCE
        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(10)
        ax.plot(T, glu211_dist, label='GYC61-GLU211')
        avrg=np.mean(glu211_dist)
        ax.plot([0, T[-1]], [avrg, avrg], ls="-", lw="0.8", c="red", label="mean={:>2.1f}".format(avrg))
        ax.set_ylabel(r'Distance ($\AA$)')
        if arg.mmd:
            # MD ANALYSIS
                ax.set_xlabel('Time (ns)')
        else:
                ax.set_xlabel('Time (fs)')
        ax.set_title(f'Distance GYC61-GLU211', fontsize=15)
        ax.legend(loc='upper right', framealpha=0.5)
        ax.set_xlim(0,T[-1])
        plt.savefig(f'GYC61_GLU211_min_dist.png', dpi=300)
        plt.close()

        # PLOT GYC61-GLU144 DISTANCE
        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(10)
        ax.plot(T, glu144_dist, label='GYC61-GLU144')
        avrg=np.mean(glu144_dist)
        ax.plot([0, T[-1]], [avrg, avrg], ls="-", lw="0.8", c="red", label="mean={:>2.1f}".format(avrg))
        ax.set_ylabel(r'Distance ($\AA$)')
        if arg.mmd:
            # MD ANALYSIS
                ax.set_xlabel('Time (ns)')
        else:
                ax.set_xlabel('Time (fs)')
        ax.set_title(f'Distance GYC61-GLU144', fontsize=15)
        ax.legend(loc='upper right', framealpha=0.5)
        ax.set_xlim(0,T[-1])
        plt.savefig(f'GYC61_GLU144_min_dist.png', dpi=300)
        plt.close()


        ###################################################################
        # GROUPING ARG88/ARG83
        fig, ax = plt.subplots(2,1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        ax[0].plot(T, all_distances[5][:], label=connections[5][0])
        ax[0].plot(T, all_distances[6][:], label=connections[6][0])
        ax[0].plot(T, all_distances[7][:], label=connections[7][0])
        ax[0].plot(T, all_distances[8][:], label=connections[8][0])
        ax[0].plot(T, all_distances[20][:], label=connections[20][0])
        ax[0].set_ylabel(r'Distance ($\AA$)')
        ax[0].set_xticklabels([])
        if arg.name:
            ax[0].set_title(f'{arg.name}: ARG88/ARG62 - GYC-OA', fontsize=20)
        else:
            ax[0].set_title(f'ARG88/ARG62 - GYC-OA/THR58', fontsize=20)
        ax[0].legend(loc='upper right', framealpha=0.5)

        ax[1].plot(T,i_torsion, label='I-torsion')
        ax[1].plot(T,p_torsion, label='P-torsion')
        ax[1].set_ylabel(r'$\phi$ (deg)')

        if arg.mmd:
            # MD ANALYSIS
            ax[1].set_xlabel('Time (ns)')
        else:
            ax[1].set_xlabel('Time (fs)')
        ax[1].legend(loc='upper left', framealpha=0.5)

        color = 'tab:green'
        ax2 = ax[1].twinx()
        ax2.plot(T, pyra, color=color, ls='--', lw=1, alpha=0.5, label='Pyr.')
        ax2.set_ylabel(r'Pyramidalization (deg)')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right', framealpha=0.5)

        ax[0].set_xlim(0,T[-1])
        ax[0].set_ylim(1.5,4.0)
        ax[1].set_xlim(0,T[-1])
        ax[1].set_ylim(-80,80)
        ax2.set_ylim(-40,40)
        

        plt.subplots_adjust(hspace=0)
        if arg.name:
            plt.savefig(f'{arg.name}_ARG88_ARG62.png', dpi=300)
        else:
            plt.savefig(f'ARG88_ARG62_GYC.png', dpi=300)
        #plt.show()
        plt.close()

        ###################################################################
        # GROUPING HIS-PRING-THR
        fig, ax = plt.subplots(2,1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        ax[0].plot(T, all_distances[9][:], label=connections[9][0])
        ax[0].plot(T, all_distances[13][:], label=connections[13][0])
        #ax[0].plot(T, all_distances[22][:], label=connections[22][0])
        ax[0].set_ylabel(r'Distance ($\AA$)')
        ax[0].set_xticklabels([])
        if arg.name:
            ax[0].set_title(f'{arg.name}: HIS190 - P-RING - THR58', fontsize=20)
        else:
            ax[0].set_title(f'HIS190 - P-RING - THR58', fontsize=20)

        ax[0].legend(loc='upper right', framealpha=0.5)

        ax[1].plot(T,i_torsion, label='I-torsion')
        ax[1].plot(T,p_torsion, label='P-torsion')
        ax[1].set_ylabel(r'$\phi$ (deg)')
        if arg.mmd:
            # MD ANALYSIS
            ax[1].set_xlabel('Time (ns)')
        else:
            ax[1].set_xlabel('Time (fs)')
        ax[1].legend(loc='upper left', framealpha=0.5)

        color = 'tab:green'
        ax2 = ax[1].twinx()
        ax2.plot(T, pyra, color=color, ls='--', lw=1, alpha=0.5, label='Pyr.')
        ax2.set_ylabel(r'Pyramidalization (deg)')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right', framealpha=0.5)

        ax[0].set_xlim(0,T[-1])
        ax[0].set_ylim(3,4.5)
        ax[1].set_xlim(0,T[-1])
        ax[1].set_ylim(-80,80)
        ax2.set_ylim(-40,40)

        plt.subplots_adjust(hspace=0)
        if arg.name:
            plt.savefig(f'{arg.name}_HIS190_P-RING_THR58.png', dpi=300)
        else:
            plt.savefig(f'HIS190_P-RING_THR58.png', dpi=300)
        plt.close()

        ###################################################################
        # GROUPING P-RING_HB
        fig, ax = plt.subplots(2,1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        ax[0].plot(T, all_distances[0][:], label=connections[0][0])
        ax[0].plot(T, all_distances[24][:], label=connections[24][0])
        if len(connections) > 25:
            ax[0].plot(T, all_distances[25][:], label=connections[25][0])
        ax[0].set_ylabel(r'Distance ($\AA$)')
        ax[0].set_xticklabels([])

        if arg.name:
            ax[0].set_title(f'{arg.name}: P-RING_HB', fontsize=20)
        else:
            ax[0].set_title(f'SER139/WAT - P-RING_HB', fontsize=20)

        ax[0].legend(loc='upper right', framealpha=0.5)

        ax[1].plot(T,i_torsion, label='I-torsion')
        ax[1].plot(T,p_torsion, label='P-torsion')
        ax[1].set_ylabel(r'$\phi$ (deg)')
        if arg.mmd:
            # MD ANALYSIS
            ax[1].set_xlabel('Time (ns)')
        else:
            ax[1].set_xlabel('Time (fs)')
        ax[1].legend(loc='upper left', framealpha=0.5)

        color = 'tab:green'
        ax2 = ax[1].twinx()
        ax2.plot(T, pyra, color=color, ls='--', lw=1, alpha=0.5, label='Pyr.')
        ax2.set_ylabel(r'Pyramidalization (deg)')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right', framealpha=0.5)

        ax[0].set_xlim(0,T[-1])
        ax[0].set_ylim(1.3,7.5)
        ax[1].set_xlim(0,T[-1])
        ax[1].set_ylim(-80,80)
        ax2.set_ylim(-40,40)

        plt.subplots_adjust(hspace=0)
        if arg.name:
            plt.savefig(f'{arg.name}_P-RING_HB.png', dpi=300)
        else:
            plt.savefig(f'P-RING_HB.png', dpi=300)

        plt.close()

        ###################################################################
        # GROUPING GLU208_HIS190_GLU141
        fig, ax = plt.subplots(2,1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        ax[0].plot(T, all_distances[18][:], label=connections[18][0])
        ax[0].plot(T, all_distances[19][:], label=connections[19][0])
        ax[0].set_ylabel(r'Distance ($\AA$)')
        ax[0].set_xticklabels([])

        if arg.name:
            ax[0].set_title(f'{arg.name}: GLU208_HIS190_GLU141', fontsize=20)
        else:
            ax[0].set_title(f'GLU208_HIS190_GLU141', fontsize=20)
        
        ax[0].legend(loc='upper right', framealpha=0.5)

        ax[1].plot(T,i_torsion, label='I-torsion')
        ax[1].plot(T,p_torsion, label='P-torsion')
        ax[1].set_ylabel(r'$\phi$ (deg)')
        if arg.mmd:
            # MD ANALYSIS
            ax[1].set_xlabel('Time (ns)')
        else:
            ax[1].set_xlabel('Time (fs)')
        ax[1].legend(loc='upper left', framealpha=0.5)

        color = 'tab:green'
        ax2 = ax[1].twinx()
        ax2.plot(T, pyra, color=color, ls='--', lw=1, alpha=0.5, label='Pyr.')
        ax2.set_ylabel(r'Pyramidalization (deg)')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right', framealpha=0.5)

        ax[0].set_xlim(0,T[-1])
        ax[0].set_ylim(1.5,2.5)
        ax[1].set_xlim(0,T[-1])
        ax[1].set_ylim(-80,80)
        ax2.set_ylim(-40,40)

        plt.subplots_adjust(hspace=0)

        if arg.name:
            plt.savefig(f'{arg.name}_GLU208_HIS190_GLU141.png', dpi=300)
        else:
            plt.savefig(f'GLU208_HIS190_GLU141.png', dpi=300)

        plt.close()

        ###################################################################
        # GROUPING PHE170_ARG88_TRP86
        fig, ax = plt.subplots(2,1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        ax[0].plot(T, all_distances[16][:], label=connections[16][0])
        ax[0].plot(T, all_distances[17][:], label=connections[17][0])
        ax[0].plot(T, all_distances[21][:], label=connections[21][0])
        ax[0].set_ylabel(r'Distance ($\AA$)')
        ax[0].set_xticklabels([])

        if arg.name:
            ax[0].set_title(f'{arg.name}: PHE170_ARG88_TRP86', fontsize=20)
        else:
            ax[0].set_title(f'PHE170_ARG88_TRP86', fontsize=20)

        ax[0].legend(loc='upper right', framealpha=0.5)

        ax[1].plot(T,i_torsion, label='I-torsion')
        ax[1].plot(T,p_torsion, label='P-torsion')
        ax[1].set_ylabel(r'$\phi_P$ (deg)')
        if arg.mmd:
            # MD ANALYSIS
            ax[1].set_xlabel('Time (ns)')
        else:
            ax[1].set_xlabel('Time (fs)')
        ax[1].legend(loc='upper left', framealpha=0.5)

        color = 'tab:green'
        ax2 = ax[1].twinx()
        ax2.plot(T, pyra, color=color, ls='--', lw=1, alpha=0.5, label='Pyr.')
        ax2.set_ylabel(r'Pyramidalization (deg)')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right', framealpha=0.5)

        ax[0].set_xlim(0,T[-1])
        ax[0].set_ylim(3,8)
        ax[1].set_xlim(0,T[-1])
        ax[1].set_ylim(-80,80)
        ax2.set_ylim(-40,40)

        plt.subplots_adjust(hspace=0)

        if arg.name:
            plt.savefig(f'{arg.name}_PHE170_ARG88_TRP86.png', dpi=300)
        else:
            plt.savefig(f'PHE170_ARG88_TRP86.png', dpi=300)
        plt.close()

        ###################################################################
        # GROUPING ILE192
        fig, ax = plt.subplots(2,1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        ax[0].plot(T, all_distances[2][:], label=connections[2][0])
        ax[0].plot(T, all_distances[3][:], label=connections[3][0])
        ax[0].plot(T, all_distances[4][:], label=connections[4][0])
        ax[0].set_ylabel(r'Distance ($\AA$)')
        ax[0].set_xticklabels([])

        if arg.name:
            ax[0].set_title(f'{arg.name}: ILE192', fontsize=20)
        else:
            ax[0].set_title(f'ILE192', fontsize=20)

        ax[0].legend(loc='upper right', framealpha=0.5)

        ax[1].plot(T,i_torsion, label='I-torsion')
        ax[1].plot(T,p_torsion, label='P-torsion')
        ax[1].set_ylabel(r'$\phi$ (deg)')
        if arg.mmd:
            # MD ANALYSIS
            ax[1].set_xlabel('Time (ns)')
        else:
            ax[1].set_xlabel('Time (fs)')
        ax[1].legend(loc='upper left', framealpha=0.5)

        color = 'tab:green'
        ax2 = ax[1].twinx()
        ax2.plot(T, pyra, color=color, ls='--', lw=1, alpha=0.5, label='Pyr.')
        ax2.set_ylabel(r'Pyramidalization (deg)')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right', framealpha=0.5)

        ax[0].set_xlim(0,T[-1])
        ax[0].set_ylim(3.5,6.0)
        ax[1].set_xlim(0,T[-1])
        ax[1].set_ylim(-80,80)
        ax2.set_ylim(-40,40)

        plt.subplots_adjust(hspace=0)

        if arg.name:
            plt.savefig(f'{arg.name}_ILE192_ALL.png', dpi=300)
        else:
            plt.savefig(f'ILE192_ALL.png', dpi=300)
        plt.close()


        r"""
        #fig, ax = plt.subplots()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        X=np.linspace(0, len(u.trajectory)-1, len(u.trajectory))

        #ax=sns.scatterplot(x=i_torsion, y=DCOM, hue=X, markers=X, alpha=0.5, palette=cmc.hawaii, sizes=X, s=100, legend=False)
        #ax=sns.kdeplot(x=i_torsion, y=DCOM, fill=True, alpha=0.5)
        ax1=sns.jointplot(x=i_torsion, y=DCOM, kind='kde', color='purple', alpha=0.5)
        ax1=sns.scatterplot(x=i_torsion, y=DCOM, hue=X, markers=X, palette=cmc.hawaii, sizes=X, s=100, legend=False, linewidth=0, alpha=0.8)

        ax1.set(xlabel=r"$\phi_P$ (deg)", ylabel=r"$R_{P-ring - HIP}$ ($\AA$)")

        #Colormap for comparison
        cmap = plt.get_cmap(cmc.hawaii)
        norm = plt.Normalize(X[0],X[-1])
        sm =  ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1)
        plt.savefig('cm.png')
        """

    if arg.distrib:
        from matplotlib.ticker import MultipleLocator
        from matplotlib import pyplot as plt
        import numpy as np
        import seaborn as sns
        import pandas as pd
        import glob


        data=[['ARG','ARG63_HH12-GYC_OA.npy','ARG63_HH22-GYC_OA.npy','ARG88_HH12-GYC_OA.npy','ARG88_HH12-THR58_O.npy','ARG88_HH22-GYC_OA.npy'],
              #['PROJ', 'PROJ_HIS190-COM_Pring-COM.npy','PROJ_PHE170-HZ_Pring-COM.npy','PROJ_THR58-CG2_COM_dist.npy','PROJ_THR58_Pring_vector.npy'],
              ['HIS','GLU141_OE2-HIS190_HE2.npy','GLU208_OE1-HIS190_HD1.npy', 'GYC61_GLU211_min_dist.npy', 'GYC61_GLU144_min_dist.npy','HIS190_COM-GYC_Pring_COM.npy', 'HIS190_COM-THR58_CG2.npy'],
              #['PHE_ARG_TRP','PHE170_CD2-ARG88_CZ.npy','PHE170_CE1-ARG88_CZ.npy','TRP86_BENZ_COM-ARG88_CZ.npy'], 
              ['ALL','ARG63_HH12-GYC_OA.npy','ARG63_HH22-GYC_OA.npy','ARG88_HH12-GYC_OA.npy','ARG88_HH12-THR58_O.npy','ARG88_HH22-GYC_OA.npy', 'SER139-HG_GYC61-OK.npy','GYC61_PHE170_min_dist.npy','ILE192_CB-GYC_CI.npy','ILE192_CG2-GYC_CI.npy','THR58_CG2-GYC_Pring_COM.npy','HIS190_COM-GYC_Pring_COM.npy', 'HIS190_COM-THR58_CG2.npy', 'GYC61_GLU211_min_dist.npy', 'GYC61_GLU144_min_dist.npy']]

        labels=[['ARG','ARG63_HH12-GYC_OA.npy','ARG63_HH22-GYC_OA.npy','ARG88_HH12-GYC_OA.npy','ARG88_HH12-THR58_O.npy','ARG88_HH22-GYC_OA.npy'],
              #['PROJ', 'PROJ_HIS190-COM_Pring-COM.npy','PROJ_PHE170-HZ_Pring-COM.npy','PROJ_THR58-CG2_COM_dist.npy','PROJ_THR58_Pring_vector.npy'],
              ['HIS','GLU141_OE2-HIS190_HE2.npy','GLU208_OE1-HIS190_HD1.npy', 'GYC61_GLU211_min_dist.npy', 'GYC61_GLU144_min_dist.npy','HIS190_COM-GYC_Pring_COM.npy', 'HIS190_COM-THR58_CG2.npy'],
              #['PHE_ARG_TRP','PHE170_CD2-ARG88_CZ.npy','PHE170_CE1-ARG88_CZ.npy','TRP86_BENZ_COM-ARG88_CZ.npy'], 
              ['ALL','ARG63_HH12-GYC_OA.npy','ARG63_HH22-GYC_OA.npy','ARG88_HH12-GYC_OA.npy','ARG88_HH12-THR58_O.npy','ARG88_HH22-GYC_OA.npy', 'SER139-HG_GYC61-OK.npy','GYC61_PHE170_min_dist.npy','ILE192_CB-GYC_CI.npy','ILE192_CG2-GYC_CI.npy','THR58_CG2-GYC_Pring_COM.npy','HIS190_COM-GYC_Pring_COM.npy', 'HIS190_COM-THR58_CG2.npy', 'GYC61_GLU211_min_dist.npy', 'GYC61_GLU144_min_dist.npy']]

        for group in data:
            distances={}
            vectors={}
            for i, dist in enumerate(group):
                if i == 0:
                    figname = dist
                else:
                    if figname == 'PROJ':
                        if 'vector' in dist:
                            name = dist.replace(".npy", "")
                            vectors[name]=np.load(dist)
                        else:
                            name = dist.replace(".npy", "")
                            distances[name]=np.load(dist)
                    else:
                        name = dist.replace(".npy", "")
                        distances[name]=np.load(dist)

            if figname == 'PROJ':
                # BOXEN PLOT
                r"""
                fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,5))
                sns.boxenplot(data=distances, width=.5, orient="h", width_method="linear", ax=ax1)
                ax1.set_xlabel(r'Distance ($\AA$)', fontsize=12)
                ax1.set_xlim(0,3)
                sns.boxenplot(data=vectors, width=.5, orient="h", width_method="linear", ax=ax2)
                ax2.set_xlabel(r'Angle (deg)', fontsize=12)
                ax2.set_xlim(-5,35)
                plt.tight_layout()
                plt.savefig(f'{figname}_dist_distrib_boxen.png', dpi=300)
                plt.close()
                """
                # BOXEN PLOT
                fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,5))
                sns.boxenplot(data=distances, width=.5, orient="h", width_method="exponential", ax=ax1)
                ax1.set_xlabel(r'Distance ($\AA$)', fontsize=12)
                ax1.set_xlim(0,3)
                sns.boxenplot(data=vectors, width=.5, orient="h", width_method="exponential", ax=ax2)
                ax2.set_xlabel(r'Angle (deg)', fontsize=12)
                ax2.set_xlim(-5,35)
                plt.tight_layout()
                plt.savefig(f'{figname}_dist_distrib_boxen2.png', dpi=300)
                plt.close()


                # VIOLIN PLOT
                r"""
                fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,5))
                sns.violinplot(data=distances, orient="h", inner="box", ax=ax1)
                ax1.set_xlabel(r'Distance ($\AA$)', fontsize=12)
                ax1.set_xlim(0,3)
                sns.violinplot(data=vectors, orient="h", inner="box", ax=ax2)
                ax2.set_xlim(-5,35)
                ax2.set_xlabel(r'Angle (deg)', fontsize=12)
                plt.tight_layout()
                #plt.show()
                plt.savefig(f'{figname}_dist_distrib_violin.png', dpi=300)
                plt.close()
                """
            else:
                    # BOXEN PLOT
                r"""
                fig, ax = plt.subplots(figsize=(10,5))
                sns.boxenplot(data=distances, width=.5, orient="h", width_method="linear")
                ax.set_xlabel(r'Distance ($\AA$)', fontsize=12)
                ax.xaxis.set_tick_params(labelsize=10)
                if figname == 'ALL':
                    ax.set_xlim(1.1,7.5)
                else:
                    ax.set_xlim(1.1,5)
                plt.tight_layout()
                plt.savefig(f'{figname}_dist_distrib_boxen.png', dpi=300)
                plt.close()
                """
                # BOXEN PLOT
                fig, ax = plt.subplots(figsize=(10,5))
                sns.boxenplot(data=distances, width=.5, orient="h", width_method="exponential")
                ax.set_xlabel(r'Distance ($\AA$)', fontsize=12)
                ax.xaxis.set_tick_params(labelsize=10)
                if figname == 'ALL' or figname == 'HIS':
                    ax.set_xlim(1.1,13)
                else:
                    ax.set_xlim(1.1,5)
                plt.tight_layout()
                #plt.show()
                plt.savefig(f'{figname}_dist_distrib_boxen2.png', dpi=300)
                plt.savefig(f'{figname}_dist_distrib_boxen2.svg', format="svg")

                plt.close()


                # VIOLIN PLOT
                r"""
                fig, ax = plt.subplots(figsize=(10,5))
                sns.violinplot(data=distances, orient="h", inner="box")
                ax.set_xlabel(r'Distance ($\AA$)', fontsize=12)
                ax.xaxis.set_tick_params(labelsize=10)
                if figname == 'ALL':
                    ax.set_xlim(1.1,7.5)
                else:
                    ax.set_xlim(1.1,5)
                plt.tight_layout()
                #plt.show()
                plt.savefig(f'{figname}_dist_distrib_violin.png', dpi=300)
                plt.close()
                """
    if arg.mindist:
        import MDAnalysis as mda
        from MDAnalysis.analysis import rdf, distances
        from matplotlib.ticker import MultipleLocator
        from matplotlib import pyplot as plt
        import numpy as np
        import seaborn as sns
        import warnings, sys, socket
        import cmcrameri.cm as cmc
        from matplotlib.cm import ScalarMappable

        if socket.gethostname() == "nhlist-desktop":
            sys.path.insert(1, '/home/rcouto/theochem/progs/tcutil/code/geom_param')
        elif socket.gethostname() == "berzelius2.nsc.liu.se":
            sys.path.insert(1, '/proj/berzelius-2023-33/users/x_rafca/progs/tcutil/code/geom_param')
        elif socket.gethostname() == "amaze":
            sys.path.insert(1, '/data/users/rcc/codes/tcutil/code/geom_param')
        else:
            sys.path.insert(1, '/Users/rafael/theochem/projects/codes/tcutil/code/geom_param') 
        import geom_param as gp
        
        # suppress some MDAnalysis warnings about PSF files
        warnings.filterwarnings('ignore')

        # LOAD TRAJECTORY
        if arg.dcd2:
            u=mda.Universe(arg.top, [arg.dcd,arg.dcd2])
        elif arg.dcdlist:
            dcds = [ file.rstrip('\n') for file in open(arg.dcdlist, 'r').readlines() ]
            u=mda.Universe(arg.top, dcds)
        else:
            u=mda.Universe(arg.top, arg.dcd)

        gyc61=u.select_atoms("resid 61")
        phe170=u.select_atoms("resid 170")
        glu208=u.select_atoms("resid 208")
        glu141=u.select_atoms("resid 141")
        ile192=u.select_atoms("resid 192")

        phe170_out=open('GYC61_PHE170_min_dist.dat', 'w')
        glu208_out=open('GYC61_GLU208_min_dist.dat', 'w')
        glu141_out=open('GYC61_GLU141_min_dist.dat', 'w')
        ile192_out=open('GYC61_ILE192_min_dist.dat', 'w')

        phe170_dist=[]
        glu208_dist=[]
        glu141_dist=[]
        ile192_dist=[]
        for _ in u.trajectory:
            d=mda.analysis.distances.distance_array(gyc61,phe170)
            min_dist=np.min(d)
            index=np.argwhere(d == min_dist)
            phe170_dist.append(min_dist)
            phe170_out.write('GYC61-{:s} PHE170-{:s}: {:>2.2f} \n'.format(gyc61[index[0][0]].name, phe170[index[0][1]].name, min_dist))

            d=mda.analysis.distances.distance_array(gyc61,glu208)
            min_dist=np.min(d)
            index=np.argwhere(d == min_dist)
            glu208_dist.append(min_dist)
            glu208_out.write('GYC61-{:s} GLU208-{:s}: {:>2.2f} \n'.format(gyc61[index[0][0]].name, glu208[index[0][1]].name, min_dist))

            d=mda.analysis.distances.distance_array(gyc61,glu141)
            min_dist=np.min(d)
            index=np.argwhere(d == min_dist)
            glu141_dist.append(min_dist)
            glu141_out.write('GYC61-{:s} GLU141-{:s}: {:>2.2f} \n'.format(gyc61[index[0][0]].name, glu141[index[0][1]].name, min_dist))

            d=mda.analysis.distances.distance_array(gyc61,ile192)
            min_dist=np.min(d)
            index=np.argwhere(d == min_dist)
            ile192_dist.append(min_dist)
            ile192_out.write('GYC61-{:s} ILE192-{:s}: {:>2.2f} \n'.format(gyc61[index[0][0]].name, ile192[index[0][1]].name, min_dist))

        phe170_out.close()
        glu208_out.close()
        glu141_out.close()
        ile192_out.close()

        np.save('GYC61_PHE170_min_dist.npy', phe170_dist)
        np.save('GYC61_GLU208_min_dist.npy', glu208_dist)
        np.save('GYC61_GLU141_min_dist.npy', glu141_dist)
        np.save('GYC61_ILE192_min_dist.npy', ile192_dist)

        if arg.mmd:
            # MD ANALYSIS
            T=np.linspace(0,len(u.trajectory),len(u.trajectory))/100
        else:            
            T=np.linspace(0,len(u.trajectory)/2,len(u.trajectory))
        # PLOT GYC61-PHE170 DISTANCE
        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(10)
        ax.plot(T, phe170_dist, label='GYC61-PHE170')
        avrg=np.mean(phe170_dist)
        ax.plot([0, T[-1]], [avrg, avrg], ls="-", lw="0.8", c="red", label="mean={:>2.1f}".format(avrg))
        ax.set_ylabel(r'Distance ($\AA$)')
        if arg.mmd:
            # MD ANALYSIS
                ax.set_xlabel('Time (ns)')
        else:
                ax.set_xlabel('Time (fs)')
        ax.set_title(f'Distance GYC61-PHE170', fontsize=15)
        ax.legend(loc='upper right', framealpha=0.5)
        ax.set_xlim(0,T[-1])
        plt.savefig(f'GYC61_PHE170_min_dist.png', dpi=300)
        plt.close()

        # PLOT GYC61-GLU208 DISTANCE
        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(10)
        ax.plot(T, glu208_dist, label='GYC61-GLU208')
        avrg=np.mean(glu208_dist)
        ax.plot([0, T[-1]], [avrg, avrg], ls="-", lw="0.8", c="red", label="mean={:>2.1f}".format(avrg))
        ax.set_ylabel(r'Distance ($\AA$)')
        if arg.mmd:
            # MD ANALYSIS
                ax.set_xlabel('Time (ns)')
        else:
                ax.set_xlabel('Time (fs)')
        ax.set_title(f'Distance GYC61-GLU208', fontsize=15)
        ax.legend(loc='upper right', framealpha=0.5)
        ax.set_xlim(0,T[-1])
        plt.savefig(f'GYC61_GLU208_min_dist.png', dpi=300)
        plt.close()

        # PLOT GYC61-GLU141 DISTANCE
        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(10)
        ax.plot(T, glu141_dist, label='GYC61-GLU141')
        avrg=np.mean(glu141_dist)
        ax.plot([0, T[-1]], [avrg, avrg], ls="-", lw="0.8", c="red", label="mean={:>2.1f}".format(avrg))
        ax.set_ylabel(r'Distance ($\AA$)')
        if arg.mmd:
            # MD ANALYSIS
                ax.set_xlabel('Time (ns)')
        else:
                ax.set_xlabel('Time (fs)')
        ax.set_title(f'Distance GYC61-GLU141', fontsize=15)
        ax.legend(loc='upper right', framealpha=0.5)
        ax.set_xlim(0,T[-1])
        plt.savefig(f'GYC61_GLU141_min_dist.png', dpi=300)
        plt.close()

        # PLOT GYC61-ILE192 DISTANCE
        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(10)
        ax.plot(T, ile192_dist, label='GYC61-ILE192')
        avrg=np.mean(ile192_dist)
        ax.plot([0, T[-1]], [avrg, avrg], ls="-", lw="0.8", c="red", label="mean={:>2.1f}".format(avrg))
        ax.set_ylabel(r'Distance ($\AA$)')
        if arg.mmd:
            # MD ANALYSIS
                ax.set_xlabel('Time (ns)')
        else:
            ax.set_xlabel('Time (fs)')
        ax.set_title(f'Distance GYC61-ILE192', fontsize=15)
        ax.legend(loc='upper right', framealpha=0.5)
        ax.set_xlim(0,T[-1])
        plt.savefig(f'GYC61_ILE192_min_dist.png', dpi=300)
        plt.close()

if __name__=="__main__":
    main()

