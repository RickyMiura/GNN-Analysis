import sys


if __name__ == '__main__':
    targets = sys.argv[1:]
    if len(targets) == 0:
        import GCN, GATv2, GIN, graphGPS
    if 'GCN' in targets:
        import GCN
    if 'GAT' in targets:
        import GATv2
    if 'GIN' in targets:
        import GIN
    if 'GPS' in targets:
        import graphGPS
    
