import os


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def update_args(args, iter_num):
    args.iter_dir = os.path.join(args.exp_dir, 'iter%d' % iter_num)
    mkdir(args.iter_dir)

    args.output_file = os.path.join(args.iter_dir, 'outputs.txt')
    args.filtered_output_file = os.path.join(args.iter_dir, 'filtered_outputs.txt')
    args.rationale_file = os.path.join(args.iter_dir, 'dual_top_rationales.txt')
    args.result_file = os.path.join(args.iter_dir, 'result.txt')
    args.model_file = os.path.join(args.iter_dir, 'model.%d' % args.epoch)
    args.save_dir = args.iter_dir


def write_list_to_file(tuple_list, file):
    with open(file, 'w') as f:
        for tup in tuple_list:
            if not isinstance(tup, str):
                f.write(' '.join(tup) + '\n')
            else:
                f.write(tup + '\n')


def save(result, args):
    write_list_to_file(result['sg_pairs'], args.output_file)
    write_list_to_file(result['pred_actives'], args.filtered_output_file)
    write_list_to_file(result['rationales'], args.rationale_file)

    # with open(args.result_file, 'w') as f:
    #     f.write('success rate: %f\n' % result['success_rate'])
    #     f.write('novelty: %f\n' % result['novelty'])
    #     f.write('diversity: %f\n' % result['diversity'])
    #     # for topk in [1, 10, 50, 100, 500, 1000]:
    #     for topk in [1, 10, 100, 1000, 5000, 10000]:
    #         if str(topk) in result:
    #             f.write(result[str(topk)])
