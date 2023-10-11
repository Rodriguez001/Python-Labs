import string
from datetime import datetime
from math import sqrt


def read_file_csv(file: str) -> dict:
    fd = open(file, 'r')
    headers = []
    pokemons = {}
    i = 0
    for line in fd.readlines():
        buffer_line = line.split(";")
        buffer_line[len(buffer_line) - 1] = buffer_line[len(buffer_line) - 1].replace("\n", "")
        if i == 0:
            headers = buffer_line
        else:
            record = {}
            for j, value in enumerate(buffer_line):
                record[headers[j]] = value
                if 4 <= j <= 10:
                    record[headers[j]] = int(value)
            pokemons[i - 1] = record
        i += 1
    return pokemons


def count_pokemons(Pokemons: dict) -> int:
    return len(Pokemons)


def is_grass(pair) -> bool:
    pok_type = 'Grass'
    key, value = pair
    return (key == 'Type 1' or key == 'Type 2') and value == pok_type


def filtrer_par_type_pokemons(pokemons: dict, pok_type: string) -> int:
    if pok_type == 'Grass':
        return count_pokemons({k: v for k, v in pokemons.items() if v['Type 1'] == pok_type or v['Type 2'] == pok_type})
    else:
        if pok_type == 'Legendary':
            return count_pokemons({k: v for k, v in pokemons.items() if v['Legendary'] == 'True'})


def puissance_totale(pokemon: dict) -> float:
    # print(pokemon)
    return pokemon['HP base'] + pokemon['Attack base'] + pokemon['Defense base'] + pokemon['Sp. Atk base'] + pokemon[
        'Sp. Def base'] + pokemon['Speed base']


def ajout_puissance(pokemons: dict) -> dict:
    # print(pokemons)
    for k, pokemon in pokemons.items():
        pokemon['Puissance tot.'] = puissance_totale(pokemon)
    return pokemons


def le_plus_puissance(pokemons: dict) -> dict:
    return max(pokemons.values(), key=lambda pokemon: pokemon["Puissance tot."])


def convertion(pokemons: dict) -> dict:
    for key, pokemon in pokemons.items():
        speedMult = 1 + (pokemon['Speed base'] - 75) / 500
        pokemon['HP Go'] = int(pokemon['HP base'] * 1.75) + 50
        pokemon['Attack Go'] = int((1 / 4 * (min(pokemon['Attack base'], pokemon['Sp. Atk base'])) + 7 / 4 * (
            max(pokemon['Attack base'], pokemon['Sp. Atk base']))) * speedMult)
        pokemon['Defense Go'] = int((3 / 4 * (min(pokemon['Defense base'], pokemon['Sp. Def base'])) + 5 / 4 * (
            max(pokemon['Defense base'], pokemon['Sp. Def base']))) * speedMult)
        pokemon['CP'] = int((sqrt(pokemon['HP Go'] * pokemon['Attack Go'] * sqrt(pokemon['Defense Go']))) / 10)
    return pokemons


def save_csv(pokemons: dict) -> str:
    filename = 'Pokemons_RPG_' + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + '.csv'
    with open(filename, 'w') as f:
        f.write(';'.join(pokemons[0].keys()))
        f.write('\n')
        for a, row in pokemons.items():
            f.write(';'.join(str(x) for x in row.values()))
            f.write('\n')
    return filename


def save_csv2(pokemons: dict) -> str:
    filename = 'Pokemons_GO_' + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + '.csv'
    with open(filename, 'w') as f:
        f.write(';'.join(pokemons[0].keys()))
        f.write('\n')
        for a, row in pokemons.items():
            f.write(';'.join(str(x) for x in row.values()))
            f.write('\n')
    return filename


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_path = "./Pokemon.csv"
    pokemons = read_file_csv(file_path)
    # print('Pokemons  :\n', pokemons)
    print(f'Combien de pokemons sont réferencés? ', count_pokemons(pokemons))
    print(f'Combien de pokemons sont de type plantes? ', filtrer_par_type_pokemons(pokemons, 'Grass'))
    print(f'Combien de pokemons sont legendaires ? ', filtrer_par_type_pokemons(pokemons, 'Legendary'))
    print('Pokemons et leur puissance dans ce fichier-->:', save_csv(ajout_puissance(pokemons)))
    print('Le pokemon le plus puissance :\n', le_plus_puissance(pokemons))
    print('Convertion Pokemon RPG => Convertion GO ici -->:', save_csv2(convertion(pokemons)))
