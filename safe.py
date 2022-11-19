# SAFE TEAM
# distributed under license: GPL 3 License http://www.gnu.org/licenses/

from asm_embedding.FunctionAnalyzerRadare import RadareFunctionAnalyzer
from argparse import ArgumentParser
from asm_embedding.FunctionNormalizer import FunctionNormalizer
from asm_embedding.InstructionsConverter import InstructionsConverter
from neural_network.SAFEEmbedder import SAFEEmbedder
from utils import utils
from sklearn.metrics.pairwise import cosine_similarity

class SAFE:

    def __init__(self, model):
        self.converter = InstructionsConverter("data/i2v/word2id.json")
        self.normalizer = FunctionNormalizer(max_instruction=150)
        self.embedder = SAFEEmbedder(model)
        self.embedder.loadmodel()
        self.embedder.get_tensor()

    def embedd_function(self, filename, address):
        analyzer = RadareFunctionAnalyzer(filename, use_symbol=False, depth=0)
        functions = analyzer.analyze()
        instructions_list = None
        for function in functions:
            print("function:",functions[function])
            if functions[function]['address'] == address:
                instructions_list = functions[function]['filtered_instructions']
                break
        if instructions_list is None:
            print("Function not found")
            return None
        converted_instructions = self.converter.convert_to_ids(instructions_list)
        instructions, length = self.normalizer.normalize_functions([converted_instructions])
        embedding = self.embedder.embedd(instructions, length)
        return embedding


if __name__ == '__main__':

    utils.print_safe()

    parser = ArgumentParser(description="Safe Embedder")

    parser.add_argument("-m", "--model",   help="Safe trained model to generate function embeddings")
    parser.add_argument("-i1", "--input1",   help="Input executable that contains the function to embedd")
    parser.add_argument("-a1", "--address1", help="Hexadecimal address of the function to embedd")
    parser.add_argument("-i2", "--input2",   help="Input executable that contains the function to embedd")
    parser.add_argument("-a2", "--address2", help="Hexadecimal address of the function to embedd")

    args = parser.parse_args()
    safe = SAFE(args.model)

    address1 = int(args.address1, 16)
    embedding1 = safe.embedd_function(args.input1, address1)
    print("embedding1", embedding1[0])

    address2 = int(args.address2, 16)
    embedding2 = safe.embedd_function(args.input2, address2)
    print("embedding2", embedding2[0])

    sim=cosine_similarity(embedding1, embedding2)
    print("sim",sim)
