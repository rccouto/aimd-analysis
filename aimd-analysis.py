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
            ts=1
           
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
    f.add_option('--dist' , action="store_true",  default = False, help='Check the closest distance between residues and the chromophore.')
    f.add_option('--name' , type = str, default = None, help='Name to be used in the output files/figures.')
    f.add_option('--spc' , type = str, default = None, help='Extract structures from trajectory for single point calculations. Module: run and get')
    f.add_option('--spcplot' ,action="store_true",  default=False, help='Extract structures from trajectory for single point calculations. Module: run and get')
    f.add_option('--test' , action="store_true",  default=False, help='For testing purposes')
    f.add_option('--h2o' , action="store_true",  default=False, help='For H20 hydrogen bonding only.')
    f.add_option('--surr' , action="store_true",  default=False, help='Gives a list with the close residues to the target in the whole trajectory.')
    f.add_option('--with2o' , action="store_false",  default=True, help='Do not exclude water in the HB analysis.')
    f.add_option('--minima' , type=str,  default=None, help='Minima analysis: "meci", "is", "ls1d" ')
    f.add_option('--meci' , action="store_true",  default=None, help='MECI analysis')
    f.add_option('-s', '--sim' , action="store_true",  default=False, help='Runs a similarities analysis.')
    f.add_option('--torsion' , action="store_true",  default=False, help='Torsion analysis.')
    f.add_option('--violin' , action="store_true",  default=False, help='Make Violing plots')
    (arg, args) = f.parse_args(sys.argv[1:])

    if len(sys.argv) == 1:
        f.print_help()
        sys.exit(1)

    # GET ENERGY MODULE
    if arg.energy:
        import numpy as np

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
            # LOAD TRAJECTORIE(S)
            topology = md.load_prmtop('sphere.prmtop')
            traj = md.load_dcd('coors.dcd', top = topology)

        elif arg.analyze == 'torsonly':
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
            import geom_param as gp
            topology = md.load_prmtop('../sphere.prmtop')
            traj = md.load_dcd('coors-all.dcd', top = topology)

        # ON BERZELIUS
        else:
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
            plt.xlim(-80,80)
            plt.ylim(-80,80) 
            #plt.ylim(-43,30)
            plt.title('P-I-Torsion')
            cbar=plt.colorbar(values=t)
            cbar.set_label('Time (fs)')
            plt.savefig('i-p-torsion-2d.png')
            plt.close()
            #plt.show(block = True)

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

        if arg.analyze == "torsonly":
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
            plt.plot(t,i_torsion, label="P-torsion")
            plt.plot(t,p_torsion, label="I-torsion")
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
        sur_resids, sur_resname = compute_neighbors(traj,chrome,0.4, True)

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

        #################
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
            plt.savefig('mim-dynam-struct.png', dpi=300)

        plt.show(block = True)
        plt.close()
        
    if arg.meci:
        import mdtraj as md 
        import socket
        import numpy as np
        from matplotlib.pyplot import cm

         # ON MACMINI    
        if socket.gethostname() == "rcc-mac.kemi.kth.se":
            sys.path.insert(1, '/Users/rafael/theochem/projects/codes/tcutil/code/geom_param') 
        # ON BERZELIUS
        else:
            sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
        import geom_param as gp


        # Gas-phase MECI structures
        gasphase=['TFHBDI-MECII-acas.xyz', 'TFHBDI-MECIP-acas.xyz', 'TFHBDI-MECIP2-acas.xyz']

        frame=[20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90]
        type=['Imax', 'Imin', 'Pmax', 'Pmin', 'PImax']

        # Chromophore indices
        chrome=[924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]

        # Related atoms
        i_pair=[22,24]
        i_triple=[21,20,18]
        p_pair=[22,21]
        p_triple=[24,27,25]

        # SETUP FIG
        fig, ax = plt.subplots()
        color = cm.Paired(np.linspace(0, 1, len(frame)))

        label=[]
        # READ OPTIMIZED DCD FILES
        for fm, c in zip(frame, color):
            I=[]
            P=[]
            for tp in type:

                topology = md.load_prmtop(f'meci-f{fm}-{tp}.prmtop')
                traj = md.load_dcd(f'meci-f{fm}-{tp}.dcd', top = topology)
                N=len(traj)-1
                # I-torsion
                teta_i = gp.compute_torsion5(traj.xyz[N,chrome,:],i_pair,i_triple)
                # P-torsion
                teta_p = gp.compute_torsion5(traj.xyz[N,chrome,:],p_pair,p_triple)

                I.append(teta_i)
                P.append(teta_p)
                print("f%d00 \t %s \t %3.2f \t %3.2f" % (fm, tp, teta_i, teta_p))
            label.append(str(f'f{fm}00'))

            scatter = ax.scatter(I,P, s=150, color=c, alpha=0.8)

        plt.legend(label, loc='upper right', frameon=False)

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
        plt.xlabel("I torsion")
        plt.ylabel("P torsion")
        plt.xlim(-200,200)
        plt.ylim(-200,200) 

        scatter = ax.scatter(Igas,Pgas, s=100, color='red', marker='D')
 
        #plt.title("MECI structures")
        #plt.savefig('meci-opt.svg', dpi=300, format='svg')


        plt.show(block = True)
        #plt.close()
        
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
                    axs[j].set_title("Radius < %2.1f $\AA$" % (dist[j]*10), y=1.0, pad=-14)
                elif j == len(dist)-1:
                    axs[j].set_title("Radius > %2.1f $\AA$" % (dist[j-1]*10), y=1.0, pad=-14)
                else:
                    axs[j].set_title("%2.1f <  Radius < %2.1f $\AA$" % (dist[j-1]*10, dist[j]*10), y=1.0, pad=-14)
        
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

        # GET STEP (fs) AND ENERGY (a.u.)
        st, step, energy, dipole = get_tc_md_results(arg.dipole)
        
        s1=(energy[:,1]-energy[:,0])*27.211385
        d1=dipole[:,0]

        time=np.linspace(0,len(s1)-1,len(s1))/2


        plt.plot(time,d1)
        plt.ylabel('Osc.strength (a.u.)')
        plt.xlabel('Time (fs)')
        plt.title('S0->S1 Oscillator strength')
        #plt.show()
        
        out=arg.dipole
        name=out.replace(".out", "-dipole.png")
        plt.savefig(name, format='png', dpi=300)


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
        plt.ylabel('$\Delta$E Protein/gas-phase (eV)  ')
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
            plt.xlabel('$\phi_I$ window (degrees)', fontsize=16)
        else:
            plt.xlabel('$\phi_P$ window (degrees)', fontsize=16)
        
        if Y == "I":
            plt.ylabel('$\phi_I$ samples (degrees)', fontsize=16)
        else:
            plt.ylabel('$\phi_P$ samples (degrees)', fontsize=16)
        
        plt.xticks(np.arange(-120, 120+1, 20), fontsize=14)
        plt.yticks(np.arange(-180, 180+1, 20), fontsize=14)
        plt.xlim([-110, 110])
        plt.ylim([-180, 180])
        #plt.axes().xaxis.set_minor_locator(MultipleLocator(10))
        #plt.axes().yaxis.set_minor_locator(MultipleLocator(10))
        plt.tight_layout()
        
        plt.savefig('violin.png', dpi=400)
        
        plt.show()

    if arg.test == True:
        import matplotlib.pyplot as plt
        import numpy as np


        Idihedrals = np.load('i_torsion_I-100.npy')
        Pdihedrals = np.load('p_torsion_I-100.npy')

        plt.plot(Pdihedrals)
        plt.show()


if __name__=="__main__":
    main()

