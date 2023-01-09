#!/usr/bin/env python3

import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import re, optparse, copy, math
import numpy as np
import mdtraj as md


def get_tc_results(file):
    """ Read output from TeraChem MD calculation and get the CASSCF energy  
    
    Params: 
        file - output from Terachem (it works with combined outputs from
        multiple/restart files, i.e., simple 'cat run01.out run02.out > run-all.out')  
    
    Result/Return:
        ts - time step
        S - array with MD steps
        E - matrix with CASSCF energies with different roots in columms 
    """
    
    E=[]
    S=[]
    ts=0
    getEnergy=0

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

        elif re.search(r"Singlet state Mulliken charges for ENUE:", i) is not None and getEnergy == 1:
             getEnergy=0

    nroots=int(len(E)/len(S))
    #print(nroots)
    E=np.array(E).reshape(-1,nroots)

    return ts, S, E

def compute_rotation(
    xyz,
    A,
    B,
    C,
    D,
    ):

    """ Compute the 360 rotation-angle property for geometry (in degrees).

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

def compute_neighbors (
  pdb,
  query_idxs,
  cutoff=0.4, #nm, 
  ):
  """
  Determine which residues are within cutoff of query indices
  """
  
  neighbor_resnames=set()
  neighbor_resids = set()
  #print('Query indices: {}'.format(query_idxs))
  #print(query_idxs)
  neighbor_atoms = md.compute_neighbors(pdb, cutoff, query_idxs)
  #print(neighbor_atoms)
  for atom_idx in neighbor_atoms[0]:
    resname =  pdb.topology.atom(atom_idx).residue
    resid =  pdb.topology.atom(atom_idx).residue.index
    neighbor_resnames.add(resname)
    neighbor_resids.add(resid)
    #print(atom_idx, resname, resid)

  #print('Neighbors within {} Angstroms'.format(cutoff*10))
  #print(neighbor_resnames)
  #for neightbor in neighbor_resnames:
  #  print(neightbor)
  #print(neighbor_resids)

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

# MAIN PROGRAM
def main():
    import sys
    f = optparse.OptionParser()
    # Get Type of Run
    f.add_option('-e', '--energy' , action="store_true",  default=False, help='Gets the CASSCF energy from TeraChem.')
    f.add_option('-o', '--out' , type = str, default = "bomd-all.out", help='TeraChem output file.')
    f.add_option('-a', '--analyze', type = str, default = None, help='Analyze trajectory from Terachem dynamics. Modules available: "hoop" (Pyramidalization angle) \n "rotation" (S-H rotation angle) ') 
    f.add_option('--hb' , action="store_true",  default=False, help='Monitor the Hydrogen bonding')
    f.add_option('--plot' , type = str,  default = None, help='Plot with matplotlib.')
    f.add_option('--dist' , action="store_true",  default = False, help='Check the closest distance between residues and the chromophore.')
    f.add_option('--name' , type = str, default = None, help='Name to be used in the output files/figures.')
    (arg, args) = f.parse_args(sys.argv[1:])

    if len(sys.argv) == 1:
        f.print_help()
        sys.exit(1)

    # GET ENERGY MODULE
    if arg.energy == True:

        # GET STEP (fs) AND ENERGY (a.u.)
        st, step, energy = get_tc_results(arg.out)
        
        # PLOT RESULS AND SAVE FIGURE
        plt.plot(step, (energy[:,1]-energy[:,0])*27.211385)
        plt.ylabel('Energy gap S1-S0 (eV)')
        plt.xlabel('Time (fs)')
        plt.title('Energy gap between S0 and S1')
        #plt.ylim(-80,80)
        plt.savefig('energy-gap-s0-s1.png')

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
        # ON MACMINI
        sys.path.insert(1, '/Users/rafael/theochem/projects/codes/tcutil/code/geom_param') 
        # ON BERZELIUS
        #sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
        import geom_param as gp

        # LOAD TRAJECTORIE(S)
        topology = md.load_prmtop('sphere.prmtop')
        # ON MACMINI
        traj = md.load_dcd('coors.dcd', top = topology)
        # ON BERZELIUS
        #traj1 = md.load_dcd('scr.coors/coors.dcd', top = topology)
        #traj2 = md.load_dcd('res01/scr.coors/coors.dcd', top = topology)
        #traj3 = md.load_dcd('res02/scr.coors/coors.dcd', top = topology)
        #traj4 = md.load_dcd('res03/scr.coors/coors.dcd', top = topology)
        #traj5 = md.load_dcd('res04/scr.coors/coors.dcd', top = topology)
        #traj6 = md.load_dcd('res05/scr.coors/coors.dcd', top = topology)
        #traj7 = md.load_dcd('res06/scr.coors/coors.dcd', top = topology)
        #traj8 = md.load_dcd('res07/scr.coors/coors.dcd', top = topology)

        #traj=mdtraj.join([traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8], discard_overlapping_frames=True)
        #del traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8

        # Chromophore indices
        chrome=[924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]

        if arg.analyze == "hoop" or arg.analyze == "all":
            # Compute and plot the teta angle for the whole trajectory
            teta_pyra=[]
            for i in range(len(traj)):
                 # Pyra Indexes: 22(C4), 23(H8), 24(C1), 21(C5) - as in the List,ChemSci2022 paper
                teta = gp.compute_pyramidalization(traj.xyz[i,chrome,:],22,23,24,21)
                teta_pyra.append(teta)

            t=np.linspace(0, len(teta_pyra)-1, len(teta_pyra))
            plt.plot(t, teta_pyra)
            plt.ylabel('HOOP (deg)')
            plt.xlabel('Time (fs)')
            plt.ylim(-43,30)
            plt.title('HOOP')
            plt.savefig('hoop.png')
            plt.close()
            #plt.show(block = True)
            

        if arg.analyze == "rotation" or arg.analyze == "all4":
            # Compute and plot the torsion angle for the whole trajectory
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

        if arg.analyze == "torsion" or arg.analyze == "all":
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

            n=len(i_torsion)
            t=np.linspace(0, len(i_torsion)-1, len(i_torsion))
            z=np.linspace(0, len(i_torsion)-1, len(i_torsion))
            alphas=np.linspace(0.8, 0.8, len(i_torsion))
            size=np.linspace(500, 50, len(i_torsion))

            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(i_torsion,p_torsion,z, c=t, cmap=cmc.hawaii, s=size, alpha=alphas, linewidth=0.1)
            plt.ylabel('P-torsion')
            plt.xlabel('I-torsion')
            plt.xlim(-80,80)
            plt.ylim(-80,80) 
            #plt.ylim(-43,30)
            plt.title('P-I-Torsion')
            #cbar=plt.colorbar()
            #cbar.set_label('Time (fs)')
            #plt.savefig('i-p-torsion-2d.png')
            #plt.close()
            plt.show(block = True)

        if arg.analyze == "distance" or arg.analyze == "all":
            
            table=read_table("closest-atoms-indexes.dat")
            #print(table)

            # Compute and plot the distance between two given atoms
            for i in range(len(table)):
                pair=np.array([[table[i,1],table[i,2]]], dtype=np.int32) # O P-ring <> Water353
                #print(pair)
                
                if arg.name:
                    fig_name = "%s-GYC-closest-dist-%s.png" % (table[i,0], arg.name)
                else:
                    fig_name = "%s-GYC-closest-dist.png" % (table[i,0])
                #print(fig_name)
                
                dist = md.compute_distances(traj,pair)

                T=np.linspace(0,len(dist)/2,len(dist))

                plt.plot(T,dist*10)
                plt.ylabel('Distance (A)')
                plt.xlabel('Time (fs)')
                if arg.name:
                    plt.title('Distance %s -- GYC (%s)' % (table[i,0], arg.name))
                else:
                    plt.title('Distance %s -- GYC' % table[i,0])
                plt.savefig(fig_name)
                plt.show(block = True)
                plt.close()




        if not arg.analyze:
            print("      Analyze module not available! \n      Rerun with -h to see the options.")
            sys.exit(1)

    if arg.dist == True:
        import mdtraj as md

        # Get the index for the residues surrounding the chromophore.
        pdb = md.load_pdb('sphere.pdb')
        chrome = pdb.topology.select('resname GYC')
        sur_resids, sur_resname = compute_neighbors(pdb,chrome)

        out = open("closest-atoms-indexes.dat", 'w')
        out.write("# Res_Name / Res_Atom_Id / GYC_Atom_Id\n")
        # Run over the residues surrounding GYC
        print("Closest atom from a residue to the chromophore\n")
        print("Residue (Element)       Index   Chromophore     Index   Dist (AA)\n------------------------------------------------------------------")
        for i in range(len(sur_resids)):
            res_ids=pdb.topology.select('resid %s' % sur_resids[i])
            #print(sur_resname[i])

            # Run over the atoms within the residue
            d=10000
            for atm in res_ids:
                for chm in chrome:
                    pair=np.array([[atm,chm]], dtype=np.int32) # P-ring <> HIP190
                    dist = md.compute_distances(pdb,pair)
                    
                    if dist < d:
                        d = dist
                        a1 = atm
                        a2 = chm

            resname=pdb.topology.atom(a1).residue
            res_element=pdb.topology.atom(a1).element
            chrm_element=pdb.topology.atom(a2).element

            t=str(resname)
            if not t.startswith("H") and d > 0.0:
                print('{:<5s}\t({:<8s})\t{:<4d}\tGYC({:<8s})\t{:<5d}\t{:>2.4f}'.format(str(resname), str(res_element), a1, str(chrm_element), a2, float(d*10)))
                out.write("%s %d %d\n" % (resname, a1, a2))
        out.close()

    if arg.hb == True:
        import mdtraj as md
        pdb = md.load_pdb('sphere.pdb')

        #hbond=md.baker_hubbard(pdb, exclude_water=False)
        hbond=md.wernet_nilsson(pdb, exclude_water=False)
        for hb in hbond[0]:
            print(pdb.topology.atom(hb[0]).residue, "----",pdb.topology.atom(hb[2]))



if __name__=="__main__":
    main()

