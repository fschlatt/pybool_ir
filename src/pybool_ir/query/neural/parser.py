from pybool_ir.query.pubmed.parser import (
    PubmedQueryParser,
    OperatorNode,
    AtomNode,
    ASTNode,
    Q,
)


class NeuralAtomNode(AtomNode):
    lucene_query: Q

    @classmethod
    def from_atom_node(cls, atom_node: AtomNode) -> "NeuralAtomNode":
        return cls(atom_node.query, atom_node.field)


class NeuralPubmedQueryParser(PubmedQueryParser):
    def parse_ast_lucene(self, raw_query: str) -> ASTNode:
        root = self.parse_ast(raw_query)
        root = self.add_lucene_queries(root)
        return root

    def add_lucene_queries(self, node: ASTNode) -> ASTNode:
        if isinstance(node, OperatorNode):
            for idx in range(len(node.children)):
                node.children[idx] = self.add_lucene_queries(node.children[idx])
            return node
        if isinstance(node, AtomNode):
            node = NeuralAtomNode.from_atom_node(node)
            node.lucene_query = self.parse_lucene(self.format(node))
            return node
        raise ValueError(f"Unknown node type {type(node)}")
