import pysplishsplash as sph
import argparse
from argparse import RawTextHelpFormatter

def get_parser():
    parser = argparse.ArgumentParser(
        description='SPlisHSPlasH', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--scene', type=str, help='scene file')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching of boundary samples/maps.')
    parser.add_argument('--no-initial-pause', action='store_true',
                        help='Disable initial pause when starting the simulation.')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI.')
    parser.add_argument('--load-fluid-pos', action='store_true',
                        help='only load fluid pos from state file')
    parser.add_argument('--load-fluid-pos-and-vel', action='store_true',
                        help='load fluid pos and vel from state file')
    parser.add_argument('--stopAt', type=float, default=-1.0,
                        help='Sets or overwrites the stopAt parameter of the scene.')
    parser.add_argument('--state-file', type=str, default='',
                        help='State file (state_<time>.bin) that should be loaded.')
    parser.add_argument('--output-dir', type=str, default='',
                        help='Output directory for log file and partio files.')
    parser.add_argument('--param', type=str, default='', help='Sets or overwrites a parameter of the scene.\n\n'
                      '- Setting a fluid parameter:\n\t<fluid-id>:<parameter-name>:<value>\n'
                      '- Example: --param Fluid:viscosity:0.01\n\n'
                      '- Setting a configuration parameter:\n\t<parameter-name>:<value>\n'
                      '- Example: --param cflMethod:1\n')
    parser.add_argument('--lr-v', type=float, default=1.0)
    parser.add_argument('--lr-omega', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--rotationWeight', type=float, default=1.)
    parser.add_argument('--positionWeight', type=float, default=1.)
    parser.add_argument('--penaltyWeight', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--maxIter', type=int, default=150)
    parser.add_argument('--num_per_checkpoint', type=int, default=20)
    parser.add_argument('--taskType', type=str, default='bottle-flip', help='Choose from: \'bottle-flip\', \'stone-skipping\', \'high-diving\',\'dambreak-bunny \' ')
    parser.add_argument('--fullDerivative', action='store_true')
    parser.add_argument('--normalizeGrad', action='store_true')
    parser.add_argument('--gradientMode', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='Choose from \'sgd\', \'adam\' ')
    # parser.add_argument('--act_rb', type=int, default=1)
    # parser.add_argument('--passive_rb', type=int, default=1)
    return parser
