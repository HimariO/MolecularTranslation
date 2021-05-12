from tokenizers import models, normalizers, pre_tokenizers, decoders, Regex, Tokenizer


vocab = {
    "[UNK]": 0,
    "[CLS]":1,
    '[SEP]': 2,
    '[PAD]': 3,
    '[MASK]': 4,
    '(': 190,
    ')': 191,
    '+': 192,
    ',': 193,
    '-': 194,
    'InChI=1S/': 195,
    '/b': 5,
    '/c': 6,
    '/h': 7,
    '/i': 8,
    '/m': 9,
    '/s': 10,
    '/t': 11,
    '0': 12,
    '1': 13,
    '10': 14,
    '100': 15,
    '101': 16,
    '102': 17,
    '103': 18,
    '104': 19,
    '105': 20,
    '106': 21,
    '107': 22,
    '108': 23,
    '109': 24,
    '11': 25,
    '110': 26,
    '111': 27,
    '112': 28,
    '113': 29,
    '114': 30,
    '115': 31,
    '116': 32,
    '117': 33,
    '118': 34,
    '119': 35,
    '12': 36,
    '120': 37,
    '121': 38,
    '122': 39,
    '123': 40,
    '124': 41,
    '125': 42,
    '126': 43,
    '127': 44,
    '128': 45,
    '129': 46,
    '13': 47,
    '130': 48,
    '131': 49,
    '132': 50,
    '133': 51,
    '134': 52,
    '135': 53,
    '136': 54,
    '137': 55,
    '138': 56,
    '139': 57,
    '14': 58,
    '140': 59,
    '141': 60,
    '142': 61,
    '143': 62,
    '144': 63,
    '145': 64,
    '146': 65,
    '147': 66,
    '148': 67,
    '149': 68,
    '15': 69,
    '150': 70,
    '151': 71,
    '152': 72,
    '153': 73,
    '154': 74,
    '155': 75,
    '156': 76,
    '157': 77,
    '158': 78,
    '159': 79,
    '16': 80,
    '161': 81,
    '163': 82,
    '165': 83,
    '167': 84,
    '17': 85,
    '18': 86,
    '19': 87,
    '2': 88,
    '20': 89,
    '21': 90,
    '22': 91,
    '23': 92,
    '24': 93,
    '25': 94,
    '26': 95,
    '27': 96,
    '28': 97,
    '29': 98,
    '3': 99,
    '30': 100,
    '31': 101,
    '32': 102,
    '33': 103,
    '34': 104,
    '35': 105,
    '36': 106,
    '37': 107,
    '38': 108,
    '39': 109,
    '4': 110,
    '40': 111,
    '41': 112,
    '42': 113,
    '43': 114,
    '44': 115,
    '45': 116,
    '46': 117,
    '47': 118,
    '48': 119,
    '49': 120,
    '5': 121,
    '50': 122,
    '51': 123,
    '52': 124,
    '53': 125,
    '54': 126,
    '55': 127,
    '56': 128,
    '57': 129,
    '58': 130,
    '59': 131,
    '6': 132,
    '60': 133,
    '61': 134,
    '62': 135,
    '63': 136,
    '64': 137,
    '65': 138,
    '66': 139,
    '67': 140,
    '68': 141,
    '69': 142,
    '7': 143,
    '70': 144,
    '71': 145,
    '72': 146,
    '73': 147,
    '74': 148,
    '75': 149,
    '76': 150,
    '77': 151,
    '78': 152,
    '79': 153,
    '8': 154,
    '80': 155,
    '81': 156,
    '82': 157,
    '83': 158,
    '84': 159,
    '85': 160,
    '86': 161,
    '87': 162,
    '88': 163,
    '89': 164,
    '9': 165,
    '90': 166,
    '91': 167,
    '92': 168,
    '93': 169,
    '94': 170,
    '95': 171,
    '96': 172,
    '97': 173,
    '98': 174,
    '99': 175,
    'B': 176,
    'Br': 177,
    'C': 178,
    'Cl': 179,
    'D': 180,
    'F': 181,
    'H': 182,
    'I': 183,
    'N': 184,
    'O': 185,
    'P': 186,
    'S': 187,
    'Si': 188,
    'T': 189,
}


def basic_tokenizer(custom_decoder=False) -> Tokenizer:
    units = [k.replace('[', '\[').replace(']', '\]') for k in vocab.keys() if len(k) > 1] + ['.']
    units = sorted(units, key=lambda x: len(x), reverse=True)
    # print("|".join(units))
    units_rex = Regex("|".join(units))

    class CustomDecoder:
        def decode(self, tokens) -> str:
            return "".join(tokens)

    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=units_rex, behavior="isolated")
    if custom_decoder:
        tokenizer.decoder = decoders.Decoder.custom(CustomDecoder())
    return tokenizer