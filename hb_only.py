import mdtraj as md
import numpy as np
import socket, sys
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

#################
## TARGET RESIDUE
target='GYC60'
chrome=[924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960]
#################

# LOAD TRAJECTORIE(S)
topology = md.load_prmtop('sphere.prmtop')

# ON MACMINI
if socket.gethostname() == "rcc-mac.kemi.kth.se":
    import hbond as hb
    #traj = md.load_dcd('coors.dcd', top = topology)
    traj = md.load_dcd('prod.dcd', top = topology)

elif socket.gethostname() == "nhlist-desktop":
    sys.path.insert(1, '/home/rcouto/theochem/progs/tcutil/code/geom_param') 
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
# ON BERZELIUS
elif socket.gethostname() == "berzelius002":
    sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
    import geom_param as gp
    import hbond as hb

    topology = md.load_prmtop('sphere.prmtop')
    
    #traj1 = md.load_dcd('scr.coors/coors.dcd', top = topology)
    #traj2 = md.load_dcd('res01/scr.coors/coors.dcd', top = topology)
    #traj=md.join([traj1,traj2], discard_overlapping_frames=True)
    #del traj1,traj2

    traj = md.load_dcd('prod.dcd', top = topology)

else:
    sys.path.insert(1, '/proj/nhlist/users/x_rafca/progs/tcutil/code/geom_param')
    import geom_param as gp
    sys.path.insert(1, '/proj/berzelius-2023-33/users/x_rafca/progs/aimd-analysis/')
    import hbond as hb

    topology = md.load_prmtop('sphere.prmtop')
    traj = md.load_dcd('coors-all.dcd', top = topology)
"""




##################################################################
##################################################################

print('-- Computing HBs')
# IDENTIFY THE HB WITH MODIFIED wernet_nilsson METHOD FROM MDTRAJ
hbond = hb.wernet_nilsson(traj, target, exclude_water=False)
print('-- DONE --')

##################################################################
##################################################################

# SET RELEVANT HB TO BE ANALYZED
relevant_hbs=['GYC60-OK']
name='OK'


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

# FOR MD ANALYSIS
t=np.linspace(0, len(hbond)-1, len(hbond))

ax1.bar(t,number_hb, width=0.1)
ax1.set_ylabel("Number of HB")
ax1.set_xticklabels([])

if name:
    np.save(f"hb_number_{name}.npy", number_hb)
else:
    np.save("hb_number_all.npy", number_hb)

# FOR MD ANALYSIS
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

r=np.linspace(0, len(hb_recept)-1, len(hb_recept))
X,Y=np.meshgrid(t,r)
Z=np.transpose(hb_recept_count)

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

ax3.set_yticks(r, list(HbRec))
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

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

relevant_hbs=['GYC60-OA','GYC60-NB']
name='Iring'


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

# FOR MD ANALYSIS
t=np.linspace(0, len(hbond)-1, len(hbond))

ax1.bar(t,number_hb, width=0.1)
ax1.set_ylabel("Number of HB")
ax1.set_xticklabels([])

if name:
    np.save(f"hb_number_{name}.npy", number_hb)
else:
    np.save("hb_number_all.npy", number_hb)

# FOR MD ANALYSIS
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

r=np.linspace(0, len(hb_recept)-1, len(hb_recept))
X,Y=np.meshgrid(t,r)
Z=np.transpose(hb_recept_count)

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

ax3.set_yticks(r, list(HbRec))
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

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

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

# FOR MD ANALYSIS
t=np.linspace(0, len(hbond)-1, len(hbond))

ax1.bar(t,number_hb, width=0.1)
ax1.set_ylabel("Number of HB")
ax1.set_xticklabels([])

if name:
    np.save(f"hb_number_{name}.npy", number_hb)
else:
    np.save("hb_number_all.npy", number_hb)

# FOR MD ANALYSIS
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

r=np.linspace(0, len(hb_recept)-1, len(hb_recept))
X,Y=np.meshgrid(t,r)
Z=np.transpose(hb_recept_count)

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

ax3.set_yticks(r, list(HbRec))
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

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################