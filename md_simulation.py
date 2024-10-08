import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
from sys import stdout

# Load PDB file and set the FFs
pdb = app.PDBFile('alanine-dipeptide-explicit.pdb')
ff = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

# Create the OpenMM MM System and ML potential
mmSystem = ff.createSystem(pdb.topology, nonbondedMethod=app.PME)
potential = MLPotential('ani1xnr')

# Choose the ML atoms
mlAtoms = [a.index for a in next(pdb.topology.chains()).atoms()]

# Create the mixed ML/MM system (we're using the nnpops implementation for performance)
mixedSystem = potential.createMixedSystem(pdb.topology, mmSystem, mlAtoms, interpolate=False, implementation="nnpops")

# Choose to run on a GPU (CUDA), with the LangevinMiddleIntegrator (NVT) and create the context
platform = mm.Platform.getPlatformByName("CUDA") 
integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.001*unit.picoseconds)

# Setup the simulation and add reporters to it
simulation = app.Simulation(pdb.topology, mixedSystem, integrator, platform)
simulation.reporters.append(app.PDBReporter('output.pdb', 1000))
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, potentialEnergy=True, temperature=True, volume=True, speed=True))

# Set the initial positions and velocities
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

# Run NVT equilibration
print("Running NVT")
simulation.step(1000)

# Run NPT production (add a baratost first)
mixedSystem.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin))
simulation.context.reinitialize(preserveState=True)
print("Running NPT")
simulation.step(10000)
