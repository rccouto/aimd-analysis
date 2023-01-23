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

  neighbor_atoms = md.compute_neighbors(pdb, cutoff, query_idxs)

  for atom_idx in neighbor_atoms[0]:
    resname =  pdb.topology.atom(atom_idx).residue
    resid =  pdb.topology.atom(atom_idx).residue.index
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
    f.add_option('--spc' , action = "store_true", default = False, help='Extract structures from trajectory for single point calculations.')
    f.add_option('--test' , action="store_true",  default=False, help='For testing purposes')
    f.add_option('--h2o' , action="store_true",  default=False, help='For H20 hydrigen bonding only.')
    f.add_option('--surr' , action="store_true",  default=False, help='Gives a list with the close residues to the target in the whole trajectory.')
    f.add_option('--with2o' , action="store_false",  default=True, help='Do not exclude water in the HB analysis.')
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
        import socket

        # ON MACMINI    
        if socket.gethostname() == "rcc-mac.kemi.kth.se":
            sys.path.insert(1, '/Users/rafael/theochem/projects/codes/tcutil/code/geom_param') 
            # LOAD TRAJECTORIE(S)
            topology = md.load_prmtop('sphere.prmtop')
            traj = md.load_dcd('coors.dcd', top = topology)

        # ON BERZELIUS
        else:
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
            import geom_param as gp

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
            plt.ylim(-43,30)
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

        if arg.analyze == "torsion" or arg.analyze == "all":
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
            plt.xlim(-80,80)
            plt.ylim(-80,80) 
            #plt.ylim(-43,30)
            plt.title('P-I-Torsion')
            cbar=plt.colorbar(values=t)
            cbar.set_label('Time (fs)')
            #plt.savefig('i-p-torsion-2d.png')
            #plt.close()
            plt.show(block = True)

        if arg.analyze == "distance":
            
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
            plt.ylim(-40,32)
            plt.title('Flapping Dihedral angle')
            plt.savefig('flap_dihedral.png')
            #plt.show(block = True)
            plt.close()


        if not arg.analyze:
            print("      Analyze module not available! \n      Rerun with -h to see the options.")
            sys.exit(1)

    if arg.dist == True:
        import mdtraj as md
        import numpy as np
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

        #print(len(traj[0]))
        # Get the index for the residues surrounding the chromophore.
        #traj = md.load_pdb('sphere.pdb')
        #print(len(pdb))
        
        # Use just the first frame for this analyis
        traj=traj[0]

        chrome = traj.topology.select('resname GYC')
        sur_resids, sur_resname = compute_neighbors(traj,chrome)

        out = open("closest-atoms-indexes.dat", 'w')
        out.write("# Res_Name / Res_Atom_Id / GYC_Atom_Id\n")
        # Run over the residues surrounding GYC
        print("Closest atom from a residue to the chromophore\n")
        print("Residue        Index   Chromophore     Index   Dist (AA)\n---------------------------------------------------------")
        for i in range(len(sur_resids)):
            #print("Get res_ids")
            res_ids=traj.topology.select('resid %s' % sur_resids[i])
            #print(sur_resname[i])

            # Run over the atoms within the residue
            d=10000
            #print("Run over res_ids", res_ids)
            for atm in res_ids:
                #print("Run over chrome ", atm)
                for chm in chrome:
                    #print("chm", chm)
                    pair=np.array([[atm,chm]], dtype=np.int32) # P-ring <> HIP190
                    dist = md.compute_distances(traj,pair)
                    
                    if dist < d:
                        d = dist
                        a1 = atm
                        a2 = chm

            resname=traj.topology.atom(a1)
            res_element=traj.topology.atom(a1).element
            chrm_element=traj.topology.atom(a2)

            t=str(resname)
            #if not t.startswith("H") and d > 0.0:
            if d > 0.0:
                #print('{:<5s}\t({:<8s})\t{:<4d}\tGYC({:<8s})\t{:<5d}\t{:>2.4f}'.format(str(resname), str(res_element), a1, str(chrm_element), a2, float(d*10)))
                print('{:<8s}\t{:<5d}\t{:<8s}\t{:<4d}\t{:>2.4f}'.format(str(resname),  a1, str(chrm_element), a2, float(d*10)))
                out.write("%s %d %d\n" % (resname, a1, a2))
        out.close()

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
        from matplotlib.lines import Line2D

        ################
        ## TARGET RESIDUE
        target='GYC60'
        #################

        # LOAD TRAJECTORIE(S)
        topology = md.load_prmtop('sphere.prmtop')

        # ON MACMINI
        if socket.gethostname() == "rcc-mac.kemi.kth.se":
            import hbond as hb
            traj = md.load_dcd('coors.dcd', top = topology)
        else:
        # ON BERZELIUS
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/aimd-analysis/')
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
        # SET UP THE ARRAY FOR THE HB COUNT
        hb_size=100
        hb_hydrogen_count = np.zeros([(len(traj)), hb_size])
        hb_recept_count = np.zeros([(len(traj)), hb_size])

        # SAVE INFO FILES 
        out = open("hydrogen-bonding.dat", 'w')
        out2 = open("hydrogen-bonding-td.dat", 'w')
        out2.write("# t (fs) / N_HB /  RES  /  H   / CHROME\n")

        print('-- Computing HBs')
        # IDENTIFY THE HB WITH wernet_nilsson METHOD FROM MDTRAJ
        hbond = hb.wernet_nilsson(traj, target, exclude_water=arg.with2o)
        print('-- DONE --')

        # SET UP ARRAYS AND SETS
        number_hb=[]
        hb_res_h=[]
        try_set=set()
        hb_recept=[]
        try_set_recept=set()
        max_n=0

        # LOOP OF THE TRAJECTORY FRAMES
        for i in range(len(traj)):
            out.write("HBs \n")
            nhb = 0
            # LOOP OVER THE IDENTIFIED HYDROGEN BONDS
            for hb in hbond[i]:
                # ONLY HB INVOLVING THE TARGET GYC60
                if str(traj.topology.atom(hb[2]).residue) == target or  str(traj.topology.atom(hb[0]).residue) == target:
                                        
                    # USE SET TO IDENTIFY UNIQUE HB RESIDUES-HYDROIGEN AND APPEND TO ARRAY
                    if traj.topology.atom(hb[1]) not in try_set:
                        try_set.add(traj.topology.atom(hb[1]))
                        hb_res_h.append(str(traj.topology.atom(hb[1])))
                    
                    # LOOP OVER THE RESIDUE-HYDROGEN 
                    for j in range(len(hb_res_h)):
                        # IF RESIDUE ON THE LIST
                        if str(traj.topology.atom(hb[1])) == str(hb_res_h[j]):    
                            hb_hydrogen_count[i][j]+=10
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
                            hb_recept_count[i][j]+=10
                            break 

                    # WRITE TO FILE
                    out2.write("%6.1f %d %s --- %s --- %s \n" % (i*0.5, nhb, traj.topology.atom(hb[0]), traj.topology.atom(hb[1]), traj.topology.atom(hb[2]) ) )
            out.write("Time = %6.1f fs, Total HB %d \n" % (i*0.5, nhb))
            number_hb.append(nhb)
        out.close()
        out2.close()

        # SET UP FIGURE SPECS
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 2, 1]})
        fig.tight_layout()
        fig.set_figheight(8)
        fig.set_figwidth(18)
        plt.subplots_adjust(hspace=0)

        # PLOT THE HB COUNTER
        t=np.linspace(0, len(hbond)-1, len(hbond))*0.5
        ax1.bar(t,number_hb)
        ax1.set_ylabel("Number of HB")
        ax1.set_xticklabels([])

        custom_xlim=(0, len(hbond)*0.5)
        plt.setp((ax1, ax2, ax3), xlim=custom_xlim)

        ### PLOT THE RESIDUE-HYDROGEN COUNTER
        # RESIZE THE hb_hydrogen_count ARRAY TO hb_res_h
        hb_hydrogen_count=np.array(hb_hydrogen_count, order='F')
        hb_hydrogen_count.resize(len(traj), len(hb_res_h))

        # SET UP THE DATA
        r=np.linspace(0, len(hb_res_h)-1, len(hb_res_h))
        X,Y=np.meshgrid(t,r)
        Z=np.transpose(hb_hydrogen_count)
        
        #PLOT 
        cmap = matplotlib.colors.ListedColormap(['#000000','blue','red'])
        ax2.set_yticks(r, list(hb_res_h))
        ax2.set_ylim(-1, len(hb_res_h)+0.4)
        ax2.set_ylabel("Residue-Hydrogen of the HB")
        ax2.set_xlabel("Time (fs)")
        ax2.scatter(X,Y,Z,c=Z, cmap=cmap, marker="o", alpha=1)
        
        ### PLOT THE HB RECEPTOR 
        # RESIZE THE hb_recept_count ARRAY TO hb_recept
        hb_recept_count=np.array(hb_recept_count, order='F')
        hb_recept_count.resize(len(traj), len(hb_recept))
        
        # SET UP THE DATA
        t=np.linspace(0, len(hbond)-1, len(hbond))*0.5
        r=np.linspace(0, len(hb_recept)-1, len(hb_recept))
        X,Y=np.meshgrid(t,r)
        Z=np.transpose(hb_recept_count)

        #PLOT 
        cmap = matplotlib.colors.ListedColormap(['#000000','blue','red','green', 'yellow'])
        legend_elements = [ Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='1 HB'),
                            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='2 HBs'),
                            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',label='3 HBs'),
                            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',label='4 HBs')]

        ax3.set_yticks(r, list(hb_recept))
        ax3.legend(handles=legend_elements, ncol=4, loc="upper left")
        ax3.set_ylabel("HB receptor")
        ax3.set_xlabel("Time (fs)")
        ax3.set_ylim(-1, len(hb_recept)+0.6)
        ax3.scatter(X,Y,Z,c=Z, cmap=cmap, marker="o", alpha=1)
        
        # SAVE FIGURE AND EXIT
        plt.savefig('hb.png')
        plt.close()
        #plt.show(block = True)

    if arg.spc == True:
        """"
        Extract a given molecule from the a QM/MM AIMD trajectory 
        and to perform single point energy calculation with TeraChem. 
        """
        import mdtraj as md
        import numpy as np
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

        # Get the index for the target molecule.
        #target = traj.topology.select('resname GYC')
        target = [924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]
        # Index of connections to be substituted by H
        subH=[908,961]
        mol=np.append(target, subH)

        # Generate XYZ files for each trajectory frame. 
        for t in range(len(traj)):
            
            if (t % 4) == 0:
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
            sur_resids, sur_resname = compute_neighbors(traj[i],chrome)
            
            for sr in sur_resname:
                if sr not in surr_res:
                    surr_res.add(str(sr))
        
        print(list(surr_res))


    if arg.test == True:
        #sys.path.insert(1, '/Users/rafael/theochem/projects/codes/mdtraj/mdtraj/geometry') 
        import hbond as hb
        import mdtraj as md 
        import numpy as np

        topology = md.load_prmtop('sphere.prmtop')
        traj = md.load_dcd('coors.dcd', top = topology)

        target='GYC60'
        hbond = hb.wernet_nilsson(traj, target, exclude_water=False)
        print(len(hbond))

        for i in range(len(traj)):
            for hb in hbond[i]:
                print("%s -- %s -- %s \n" % (traj.topology.atom(hb[0]), traj.topology.atom(hb[1]), traj.topology.atom(hb[2]) ) )

if __name__=="__main__":
    main()

