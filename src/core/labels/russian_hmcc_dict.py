# Dict for Handwritten Mongolian Cyrillic Characters (HMCC) database
# https://www.kaggle.com/vimpigro/handwritten-mongolian-cyrillic-characters-database/data#HMCC%20similar%20merged.csv


def prepare_dict():
    comparator_dict = {}
    base = ord('А')
    end = ord('Я') + 1
    offset = end - base
    additional_index_offset = 0
    for i in range(base, end):
        current_char = chr(i)
        base_index = (i - base) * 2 + additional_index_offset
        if current_char not in ('Й', 'Ъ', 'Ы', 'Ь'):
            comparator_dict[base_index] = chr(i)
        else:
            additional_index_offset -= 1
            base_index -= 1
        comparator_dict[base_index + 1] = chr(i + offset)

        if current_char == 'Е':
            comparator_dict[base_index + 2] = 'Ё'
            comparator_dict[base_index + 3] = 'ё'
            additional_index_offset += 2
        elif current_char == 'О':
            comparator_dict[base_index + 2] = 'Ө'
            comparator_dict[base_index + 3] = 'ө'
            additional_index_offset += 2
        elif current_char == 'У':
            comparator_dict[base_index + 2] = 'Ү'
            comparator_dict[base_index + 3] = 'ү'
            additional_index_offset += 2

    return comparator_dict


COMPARATOR_DICT = prepare_dict()


def letter_from_label(class_label: int) -> chr:
    return COMPARATOR_DICT[class_label]
