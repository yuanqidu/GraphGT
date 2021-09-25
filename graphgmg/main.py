from train import *
import argparse
import json


if __name__ == '__main__':
    # All necessary arguments are defined in args.py

    args = Args()

    parser = argparse.ArgumentParser(description='123')
    parser.add_argument('--extra_arg', default='args.py', help='extra arg file path')

    with open(parser.parse_args().extra_arg, 'r') as f:
        extra_args = json.load(f)

    input_root_dir = '../waxman/WAXMAN_full/timestep_0'
    # dataset_name = extra_args['dataset']
    # timestep_name = extra_args['timestep']

    dataset_name = 'waxman'
    args.graph_type = dataset_name
    
    args.epochs = 3000

    args.max_num_node = extra_args['max_num_node']
    args.max_prev_node = extra_args['max_prev_node']

    args.model_save_path = os.path.join('./model_save', args.note.lower(), dataset_name.lower(), '')
    args.graph_save_path = os.path.join('./graphs', args.note.lower(), dataset_name.lower(), '')
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.exists(args.graph_save_path):
        os.makedirs(args.graph_save_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix',args.fname)
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run"+time, flush_secs=5)

    # load raw dataset
    # graphs = create_graphs.create(args)

    # split datasets
    # random.seed(123)
    # shuffle(graphs)
    # graphs_len = len(graphs)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8*graphs_len)]
    # graphs_validate = graphs[0:int(0.2*graphs_len)]

    # if use pre-saved graphs

    graphs_train = load_graph_list(os.path.join(input_root_dir, 'train.dat'), is_real=True)
    graphs_test = load_graph_list(os.path.join(input_root_dir, 'test.dat'), is_real=True)
    graphs_validate = load_graph_list(os.path.join(input_root_dir, 'test.dat'), is_real=True)

    graph_validate_len = 0
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
    graph_validate_len /= len(graphs_validate)
    print('graph_validate_len', graph_validate_len)

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)
    print('graph_test_len', graph_test_len)

    # max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    # min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    args.max_num_node = max([graphs_train[i].number_of_nodes() for i in range(len(graphs_train))])
    max_num_edge = max([graphs_train[i].number_of_edges() for i in range(len(graphs_train))])
    min_num_edge = min([graphs_train[i].number_of_edges() for i in range(len(graphs_train))])

    # args.max_num_node = 2000
    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(len(graphs_train),len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge,min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    # save ground truth graphs
    ## To get train and test set, after loading you need to manually slice
    save_graph_list(graphs_train, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs_test, args.graph_save_path + args.fname_test + '0.dat')
    print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

    ### comment when normal training, for graph completion only
    # p = 0.5
    # for graph in graphs_train:
    #     for node in list(graph.nodes()):
    #         # print('node',node)
    #         if np.random.rand()>p:
    #             graph.remove_node(node)
        # for edge in list(graph.edges()):
        #     # print('edge',edge)
        #     if np.random.rand()>p:
        #         graph.remove_edge(edge[0],edge[1])


    ### dataset initialization
    if 'nobfs' in args.note:
        print('nobfs')
        dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.max_num_node)
        args.max_prev_node = args.max_num_node-1
    if 'barabasi_noise' in args.graph_type:
        print('barabasi_noise')
        dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train,max_prev_node=args.max_prev_node)
        args.max_prev_node = args.max_num_node - 1
    else:
        dataset = Graph_sequence_sampler_pytorch(graphs_train,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.batch_size*args.batch_ratio, replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                               sampler=sample_strategy)

    ### model initialization
    ## Graph RNN VAE model
    # lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
    #                   hidden_size=args.hidden_size, num_layers=args.num_layers).cuda()

    if 'GraphRNN_VAE_conditional' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).cuda()
        output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_MLP' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).cuda()
        output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_RNN' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=True, output_size=args.hidden_size_rnn_output).cuda()
        output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                           has_output=True, output_size=1).cuda()

    ### start training
    train(args, dataset_loader, rnn, output)

    ### graph completion
    # train_graph_completion(args,dataset_loader,rnn,output)

    ### nll evaluation
    # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter = 200, graph_validate_len=graph_validate_len,graph_test_len=graph_test_len)

