from multiobj_rationale.properties import get_scoring_function


props_short = ['gsk3', 'jnk3']
props_long = ['gsk3', 'jnk3', 'qed', 'sa']

def get_properties(sg_pairs, all=True):
    """Get property scores for molecules

    :param sg_pairs: pairs of (rationale, generated molecule)
    :param all: use all 4 properties if all, otherwise 2
    :return: tuples of (rationale, molecule, score_1, score_2, ...)
    """
    props = props_long if all else props_short
    funcs = [get_scoring_function(prop) for prop in props]

    all_x, all_y = zip(*sg_pairs)
    props = [func(all_y) for func in funcs]

    col_list = [all_x, all_y] + props
    sgs_tuples = list(zip(*col_list))

    return sgs_tuples