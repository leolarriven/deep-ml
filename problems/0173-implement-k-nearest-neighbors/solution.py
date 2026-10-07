import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    points = np.array(points)
    query_point = np.array(query_point)
    # Source - https://stackoverflow.com/a/1401828
# Posted by u0b34a0f6ae, modified by community. See post 'Timeline' for change history
# Retrieved 2026-10-07, License - CC BY-SA 4.0

    dist = np.linalg.norm(points - query_point, axis = 1)
    dist = np.argsort(dist, kind  = 'stable')

    return [tuple(point) for point in points[dist[:k]]]
