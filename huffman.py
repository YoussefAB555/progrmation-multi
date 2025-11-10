import sys
import heapq
from collections import Counter


# --- Classe Node ---
# Nickel, cette classe est parfaite.
# Elle stocke le symbole (ex: 'a'), sa fréquence, et ses enfants.
# Le __lt__ (less than) est crucial pour que heapq sache
# comment trier les nœuds : toujours par la fréquence la plus basse.
class Node:
    def __init__(self, symbol=None, frequency=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        # Définit la comparaison pour le 'heapq' (file de priorité)
        return self.frequency < other.frequency


# J'ai renommé la fonction pour qu'elle décrive mieux ce qu'elle fait
def build_huffman_tree(frequency_table):
    """Construit l'arbre de Huffman à partir de la table de fréquences."""

    # 1. Crée une file de priorité (une liste) avec toutes nos feuilles (nœuds de caractères)
    priority_queue = [Node(char, f) for char, f in frequency_table.items()]

    # 2. Transforme cette liste en un "tas" (min-heap),
    #    ce qui permet de retirer l'élément le plus petit (plus basse fréq) très vite.
    heapq.heapify(priority_queue)

    # 3. Boucle tant qu'on n'a pas tout fusionné en un seul arbre (il reste plus d'1 nœud)
    while len(priority_queue) > 1:
        # 4. Retire les DEUX nœuds avec les plus petites fréquences
        left_child = heapq.heappop(priority_queue)
        right_child = heapq.heappop(priority_queue)

        # 5. Calcule la fréquence du nouveau nœud parent
        merged_freq = left_child.frequency + right_child.frequency
        # 6. Crée le nœud parent (sans symbole, car c'est une branche)
        merged_node = Node(frequency=merged_freq)

        # --- CORRECTION 1 (BUG MAJEUR) ---
        # Il manquait ces deux lignes !
        # Sans elles, tu crées un nœud parent qui "oublie" ses enfants.
        # L'arbre ne se construit pas, et la génération de code échoue.
        # On dit au parent qui sont ses enfants (le '0' est à gauche, '1' à droite).
        merged_node.left = left_child
        merged_node.right = right_child
        # --- Fin Correction 1 ---

        # 7. Remet le nouveau nœud fusionné dans la file.
        # Il sera trié automatiquement par heapq grâce au __lt__ de la classe Node.
        heapq.heappush(priority_queue, merged_node)

    # 8. À la fin, il ne reste qu'un seul élément : la racine de l'arbre.
    return priority_queue[0]


def generate_huffman_codes(node):
    """Génère les codes (ex: 'a': '010') en parcourant l'arbre."""

    # Dictionnaire final pour stocker les codes
    huffman_codes = {}

    # On utilise une fonction "interne" (récursive) pour parcourir l'arbre
    # Elle garde en mémoire le code binaire "en cours de construction" (ex: "01", "011"...)
    def _traverse(current_node, code=""):

        # --- CORRECTION 2 (BUG MAJEUR) ---
        # La condition était 'is None', ce qui est l'inverse.
        # On ne veut travailler que si le nœud EXISTE (n'est PAS None).
        if current_node is not None:

            # --- CORRECTION 3 (Logique de récursion) ---
            # On doit tester SI c'est une FEUILLE ou une BRANCHE.

            # CAS 1 : C'est une FEUILLE (elle a un symbole, ex: 'a')
            if current_node.symbol is not None:
                # On a atteint le bout ! On stocke le code pour ce symbole.
                # Si le code est vide (cas d'un fichier avec un seul caractère), on met '0'
                huffman_codes[current_node.symbol] = code if code else "0"

            # CAS 2 : C'est une BRANCHE (un nœud interne, symbol is None)
            else:
                # Ce n'est pas une feuille, donc on continue la récursion.
                # On va à gauche en ajoutant "0" au code
                _traverse(current_node.left, code + "0")
                # On va à droite en ajoutant "1" au code
                _traverse(current_node.right, code + "1")

        # (Si current_node is None, la fonction s'arrête juste,
        #  ce qui stoppe la récursion quand on tombe d'une feuille)
        # --- Fin Corrections 2 & 3 ---

    # On lance la récursion en commençant par la racine (le 'node' donné)
    _traverse(node)
    return huffman_codes


# --- Point d'entrée du script ---
if __name__ == "__main__":

    # 1. Vérifier si le nom du fichier est fourni
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <nom_du_fichier.txt>")
        sys.exit(1)  # Quitte le script s'il n'y a pas le bon argument

    filename = sys.argv[1]

    # 2. Lire le fichier
    try:
        with open(filename, 'r', encoding='latin-1') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{filename}' n'a pas été trouvé.")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        sys.exit(1)

    if not text:
        print("Erreur : Le fichier est vide.")
        sys.exit(1)

    print(f"--- Analyse du fichier : {filename} ---")

    # 3. Construire la table de fréquences
    # Counter fait ça très bien, c'est un dictionnaire {'char': count}
    freq_table = Counter(text)

    # 4. Construire l'arbre de Huffman
    root = build_huffman_tree(freq_table)

    # 5. Générer les codes Huffman
    huffman_codes = generate_huffman_codes(root)

    # 6. Afficher les rapports (comme demandé dans le TP)

    # --- Rapport 1 : Tableau des fréquences ---
    print("\n--- 1. Tableau des fréquences (20 plus fréquents) ---")
    # On trie le dictionnaire par fréquence (la valeur, d'où item[1])
    sorted_freq = sorted(freq_table.items(), key=lambda item: item[1], reverse=True)

    for char, freq in sorted_freq[:20]:
        print(f"{repr(char):<6} : {freq}")  # repr() pour bien afficher '\n' au lieu de sauter une ligne

    # Affiche un message si on n'a pas tout montré
    if len(freq_table) > 20:
        print(f"(... et {len(freq_table) - 20} autres caractères)")

    # --- Rapport 2 : Table de correspondance ---
    print("\n--- 2. Table de correspondance (20 plus fréquents) ---")

    # --- AJOUT D'UN CHECK DE SÉCURITÉ ---
    # Si le dictionnaire est vide, c'est que la génération de code a échoué.
    if not huffman_codes:
        print("Erreur : La génération des codes a échoué. L'arbre est peut-être mal construit.")
    else:
        for char, freq in sorted_freq[:20]:
            # On utilise .get() pour éviter un crash si un caractère n'a pas de code
            # (ce qui ne devrait pas arriver maintenant, mais c'est plus sûr)
            code = huffman_codes.get(char, "CODE_NON_TROUVE")
            print(f"{repr(char):<6} : {code}")

        if len(huffman_codes) > 20:
            print(f"(... et {len(huffman_codes) - 20} autres codes)")

    # --- Rapport 3 : Compte-rendu des tailles ---
    print("\n--- 3. Compte-rendu de compression ---")

    total_chars = len(text)
    # Taille originale (supposant 8 bits par caractère, comme ASCII/latin-1)
    original_size_bits = total_chars * 8
    compressed_size_bits = 0

    # --- AJOUT D'UN CHECK DE SÉCURITÉ ---
    # On vérifie que les codes existent avant de calculer
    if not huffman_codes:
        print("Calcul de la taille compressée impossible.")
    else:
        try:
            # Calcule la taille totale : somme de (fréq * longueur du code) pour chaque caractère
            for char, freq in freq_table.items():
                compressed_size_bits += freq * len(huffman_codes[char])

            # Calcule le ratio (si le fichier n'est pas vide)
            if original_size_bits > 0:
                compression_ratio = 1 - (compressed_size_bits / original_size_bits)

                print(f"Nombre total de caractères : {total_chars}")
                print(f"Taille originale (sur 8 bits/char) : {original_size_bits} bits")
                print(f"Taille encodée (Huffman) :         {compressed_size_bits} bits")
                print("--------------------------------------------------")
                print(f"Taux de compression : {compression_ratio:.2%}")
                print(f"Bits économisés :     {original_size_bits - compressed_size_bits} bits")
            else:
                print("Le fichier est vide, aucun calcul de compression.")

        except KeyError as e:
            # Sécurité si un caractère du texte n'a pas de code (ne devrait plus arriver)
            print(f"Erreur fatale: Le caractère {e} est dans le texte mais n'a pas de code Huffman!")
            print("Le calcul de la compression a échoué.")