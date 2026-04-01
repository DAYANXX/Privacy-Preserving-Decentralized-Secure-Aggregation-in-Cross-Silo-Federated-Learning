# To run SVFL.py, you need to run g_caculation.py first to accelerate SVFL's signature calculation


from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hashlib
from models.Nets import MLP
import torch
import os
import numpy as np
import time
import pickle

def generate_seeds(num_clients):
    """Generate random seeds for the clients."""
    return [os.urandom(16) for _ in range(num_clients)]  # 16 bytes for each seed


def generate_rsa_keys():
    """Generate RSA public and private keys."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    return private_key, public_key


def generate_g_values(num_g):
    """Generate random g values, here simply using random numbers."""
    return [int.from_bytes(os.urandom(2), 'big') for _ in range(num_g)]


def load_squares(filename):
    with open(filename, 'rb') as f:
        squares = pickle.load(f)
    return squares

def generate_signature(g, w, squares, N):
    g = g
    product = 1
    bit_index = 0
    if w > 0:
        bit_index = int(w / 1000)
        res = w % 1000
        product = pow(g, res, N)
    result = squares[g][bit_index] * product % N
    return result


def sign(i, fid, state_dict, private_key, g_values):
    squares = load_squares('precomputed_squares.pkl')
    N = 27853941290515214512788269701478767696344214848919501748748787300059281042828499385358055209711997860919340119737495690157469232384556394904674074801046277680699848319687172895073581927112527725985327442791446344962774102882947982393340311503675259218428397633040219602437280995309968666075529303632143390971347543248635240042940221966624239960261512672432939485596227583326198726926196205430568399863703469003006550558633128073802183646956460195579433276509941938416325057853889409097321205302825247425245759061862058582849529824573775739192198746447654322507287307353777816581622396328894496525602734523318407848733
    d = 13791604664193031584295578860994339254869306984565021770097779084897442205773605826553990746527218679323627674221916400592182684451208554170265250580495776595491250407767349137817381533100409306318932443025808839190716994042322078042386638208863117958374681526346264340740189027538771735266352227029968613714533591770761117541788989900456773536493084264483705984451712992120999783900129876506097090855374671800180055986624855496059255484603975763737617864048756072525908696634825245436691076183686640925238353892713846723169222824453847232692093977726296065387825996982498775441335854829530780932958733057262388905273

    hasher = hashlib.sha256()
    hasher.update(f'{i}{fid}'.encode())
    hash_value = int(hasher.hexdigest(), 16)

    product = 1
    param_keys = list(state_dict.keys())
    for j, key in enumerate(param_keys):
        param = state_dict[key]
        param_flat = param.flatten().numpy()
        g_value = g_values[j]
        for value in param_flat:
            scaled_value = int(value * 1e6)
            term = generate_signature(g_value, scaled_value, squares, N)
            product = product * term
            product %= N

    sigma = (pow(hash_value * product, d, N))
    return sigma


# Define number of clients
num_clients = 100
print('num_clients: ', num_clients)
# Initialize a simple network and get its state_dict
net = MLP(dim_in=784, dim_hidden=200, dim_out=10)
state_dict = net.state_dict()
total_bytes = sum(p.nelement() * p.element_size() for p in state_dict.values())
total_kb = total_bytes / 1024
print(f"Total KB: {total_kb} KB")

num_g = sum(p.numel() for p in net.parameters() if p.requires_grad)

seeds = generate_seeds(num_clients)
private_key, public_key = generate_rsa_keys()
N = 27853941290515214512788269701478767696344214848919501748748787300059281042828499385358055209711997860919340119737495690157469232384556394904674074801046277680699848319687172895073581927112527725985327442791446344962774102882947982393340311503675259218428397633040219602437280995309968666075529303632143390971347543248635240042940221966624239960261512672432939485596227583326198726926196205430568399863703469003006550558633128073802183646956460195579433276509941938416325057853889409097321205302825247425245759061862058582849529824573775739192198746447654322507287307353777816581622396328894496525602734523318407848733
d = 13791604664193031584295578860994339254869306984565021770097779084897442205773605826553990746527218679323627674221916400592182684451208554170265250580495776595491250407767349137817381533100409306318932443025808839190716994042322078042386638208863117958374681526346264340740189027538771735266352227029968613714533591770761117541788989900456773536493084264483705984451712992120999783900129876506097090855374671800180055986624855496059255484603975763737617864048756072525908696634825245436691076183686640925238353892713846723169222824453847232692093977726296065387825996982498775441335854829530780932958733057262388905273
e = 65537
g_values = generate_g_values(num_g)
g_values = [3, 5, 7, 11]
fid = os.urandom(32)
IV = os.urandom(16)
masked_state_dict = []
total_masks = {}
per_client_time = 0
time1 = time.time()
for j in range(num_clients):
    for name, param in state_dict.items():
        shape = param.shape
        num_bytes = param.numel()

        total_mask = torch.zeros(param.size(), dtype=torch.float32)

        seed = seeds[j]

        cipher = Cipher(algorithms.AES(seed), modes.CTR(IV),
                        backend=default_backend())
        encryptor = cipher.encryptor()
        random_bytes = encryptor.update(b'\x00' * num_bytes) + encryptor.finalize()

        random_floats = np.frombuffer(random_bytes, dtype=np.uint8) / 255.0
        random_tensor = torch.tensor(random_floats, dtype=torch.float32).view(shape)
        total_mask += random_tensor

        total_masks[name] = total_mask

    masked_state_dict.append({name: param + total_masks[name] for name, param in state_dict.items()})


signature = []
for i in range(num_clients):
    signature.append(sign(i, fid, masked_state_dict[i], private_key, g_values))

time2 = time.time()
per_client_time += (time2 - time1) / num_clients

time3 = time.perf_counter()
# Aggregation
result_state_dict = {key: torch.zeros_like(next(iter(masked_state_dict))[key]) for key in masked_state_dict[0]}
for state_dict in masked_state_dict:
    for key in result_state_dict:
        result_state_dict[key] += state_dict[key]

combine_signature = 1
for sigma in signature:
    combine_signature *= sigma
    combine_signature = combine_signature % N
time4 = time.perf_counter()

server_time = time4 - time3
time4 = time.time()


# Verification
combine_hash_value = 1
hasher = hashlib.sha256()
for i in range(num_clients):
    hasher.update(f'{i}{fid}'.encode())
    hash_value = int(hasher.hexdigest(), 16)
    combine_hash_value *= hash_value
    combine_hash_value = combine_hash_value % N

public_numbers = public_key.public_numbers()
squares = load_squares('precomputed_squares.pkl')

product = 1
param_keys = list(result_state_dict.keys())
for j, key in enumerate(param_keys):
    param = result_state_dict[key]
    param_flat = param.flatten().numpy()
    g_value = g_values[j]
    for value in param_flat:
        scaled_value = int(value * 1e6)
        term = generate_signature(g_value, scaled_value, squares, N)
        product = product * term
        product %= N

sigma = (pow(combine_hash_value * product, 1, N))
verify = pow(combine_signature, e, N)

time5 = time.time()

per_client_time += (time5 - time4)

print('per client time: ', per_client_time * 1000, 'ms')
print('server time:', server_time * 1000, 'ms')
