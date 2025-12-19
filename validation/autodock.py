import os
from vina import Vina
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

class AutoDockVinaRunner:
    def __init__(self, receptor_file, ligand_file):
        self.receptor_file = receptor_file
        self.ligand_file = ligand_file
        self.vina_instance = None
        
    def prepare_receptor(self, receptor_pdbqt):
        """Load and prepare the receptor protein"""
        self.vina_instance = Vina(sf_name='vina')
        self.vina_instance.set_receptor(receptor_pdbqt)
        
    def prepare_ligand(self, ligand_pdbqt):
        """Load and prepare the ligand"""
        self.vina_instance.set_ligand_from_file(ligand_pdbqt)
        
    def define_search_space(self, center_x, center_y, center_z, 
                           size_x=20, size_y=20, size_z=20):
        """Define the binding site search space"""
        self.vina_instance.compute_vina_maps(
            center=[center_x, center_y, center_z],
            box_size=[size_x, size_y, size_z]
        )
        
    def run_docking(self, exhaustiveness=8, n_poses=9):
        """Execute the docking simulation"""
        self.vina_instance.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        return self.vina_instance.energies(n_poses=n_poses)
        
    def save_results(self, output_file):
        """Save docking results to file"""
        self.vina_instance.write_poses(output_file, n_poses=9, overwrite=True)

# Example usage
def main():
    # Initialize the docking runner
    docker = AutoDockVinaRunner("receptor.pdbqt", "ligand.pdbqt")
    
    # Prepare receptor and ligand
    docker.prepare_receptor("receptor.pdbqt")
    docker.prepare_ligand("ligand.pdbqt")
    
    # Define search space (coordinates of binding site)
    docker.define_search_space(
        center_x=15.0, center_y=10.0, center_z=5.0,
        size_x=20, size_y=20, size_z=20
    )
    
    # Run docking
    energies = docker.run_docking(exhaustiveness=16, n_poses=20)
    
    # Save results
    docker.save_results("docking_results.pdbqt")
    
    # Print binding energies
    print("Binding energies (kcal/mol):")
    for i, energy in enumerate(energies):
        print(f"Pose {i+1}: {energy[0]:.2f}")

if __name__ == "__main__":
    main()
