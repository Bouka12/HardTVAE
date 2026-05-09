"""
code to 
*   fuse the multi view fidelity results and 
*   implement the statistical validity test 
*   This hopefully can be done later after getting the results >:<

HARMONIC MEAN is used for calculating a single unified Multi-View Fidelity Index (MFI)

Input:
    - from a dict of fidelity results for the four views, e.g. {v_dist: 0.1, v_hard: 0.2, v_comp: 0.3, v_topo: 0.4}
Task:
    - calculate the MFI using the harmonic mean of the four views fused_fidelity = 4 / (1/v_dist + 1/v_hard + 1/v_comp + 1/v_topo)
Output:
    - a single unified Multi-View Fidelity Index (MFI) value

Prior to the fusion, we need to adapt the topological fidelity to be in the similarity space [0,1] because in its current form is a distance [0, inf). 
    - applying a transformation such as: v_topo_similarity = 1 / (1 + v_topo_distance) 
"""