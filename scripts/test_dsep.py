import networkx as nx
from networkx.algorithms.d_separation import is_d_separator
from pgmpy.base import DAG

G = nx.DiGraph([('A', 'B'), ('B', 'C')])
print("networkx is_d_separator:", is_d_separator(G, {'A'}, {'C'}, {'B'}))

p_dag = DAG(list(G.edges()))
print("pgmpy not is_dconnected:", not p_dag.is_dconnected('A', 'C', observed={'B'}))
