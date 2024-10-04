# MicroSim Installation: Singularity and Grand-Potential solver

1. Install singularity from GitHub according to your OS-version:

    a. Ubuntu 18.04:
    
    > cd ~/
        
    > wget https://github.com/sylabs/singularity/releases/download/v3.11.3/singularity-ce_3.11.3-bionic_amd64.deb
        
    > sudo apt install -y ./singularity-ce_3.11.3-bionic_amd64.deb

    b. Ubuntu 20.04:
     
    > cd ~/
     	
    > wget https://github.com/sylabs/singularity/releases/download/v3.11.4/singularity-ce_3.11.4-focal_amd64.deb
     	
    > sudo apt install -y ./singularity-ce_3.11.4-focal_amd64.deb

    c. Ubuntu 22.04:
     
    > cd ~/
     	
    > wget https://github.com/sylabs/singularity/releases/download/v3.11.4/singularity-ce_3.11.4-jammy_amd64.deb
     	
    > sudo apt install -y ./singularity-ce_3.11.4-jammy_amd64.deb
        

2. Download the MicroSim package from GitHub repository: 
   [https://github.com/ICME-India/MicroSim/archive/refs/heads/main.zip](https://github.com/ICME-India/MicroSim/archive/refs/heads/main.zip)
   
   > cd ~/
   
   > wget https://github.com/ICME-India/MicroSim/archive/refs/heads/main.zip
   
   > unzip main.zip

3. If you have a [pre-built singularity image](https://drive.google.com/file/d/1TupV1wdxpbEI_KssUNLTt945QX9R1U10/view) then skip to step 4. If you do not have a pre-built image, then to build singularity image, go to the folder where def file is located, then run the following command in the terminal:

   > sudo singularity build GP_GUI.sif MicroSim-main/def_files/GP_GUI.def

This will generate a GP_GUI.sif file. Move this file to a folder which contains MicroSim

4. Run the following command in terminal (so that Grand-Potential solver can be used via GUI):

   > mv MicroSim-main MicroSim

   > cd MicroSim

   > sh swap_script.sh

   > cd ..

5. To load the image in singularity and open singularity terminal, following command has to be run from the folder which contains both GP_GUI.sif and MicroSim:

   > singularity shell --bind /run/user,./MicroSim:/mnt GP_GUI.sif

6. Go to the MicroSim folder using the following command:

   > cd /mnt

7. Running MicroSim from terminal:

* To open the GUI

  > python3 MicroSim.py

* To run the Grand potential solver, go to directory

  > cd Grand_potential_Finite_difference_2D_MPI

* clean the old compiled files

  > make clean

* compile the solver

  > make

* Run the solver on 4 cores

  > mpirun -np 4 ./microsim_gp Input_tdb_new.in Filling.in outputname 2 2    


# Using in Windows 10 and 11 with WSL2

1. Enable WSL from settings. Install WSL2 in Windows

* Open Command Prompt in administrator Mode:

  > wsl –install

  > wsl --update

2. Reboot the system

3. From Microsoft stores install Ubuntu-20.04

4. Open Ubuntu-20.04

5. Follow the steps from the previous section

