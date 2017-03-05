require 'paths'
require 'pl'
utils = paths.dofile('utils/utils.lua')

local options = {}

options.O = OrderedMap()
local mp = OrderedMap()
options.O['model'] = mp
mp.rnn_size = {default=50,
    comment='size of the hidden layer in the controller'}
mp.num_lstm_layers = {default=1,
    comment='number of lstm layers to use in the controller'}
mp.model = {default='lstm',
    comment='lstm | lantm | slantm | ram'}
mp.init_magnitude = {default=false,
    comment='if false, then no reinitialization of weights ' ..
            '(i.e. initialized according to nn module defaults). ' ..
            'Otherwise, using nn.Linear:reset to reinitialize weights'}
mp.embedding_size = {default=14,
    comment='size of integer encoded embeddings'}
mp.forget_bias = {default=1,
    comment='LSTM forget gate initialization. number | "init_magnitude"'}
mp.in_bias = {default='init_magnitude',
    comment='LSTM in gate initialization. number | "init_magnitude"'}
mp.out_bias = {default='init_magnitude',
    comment='LSTM out gate initialization. number | "init_magnitude"'}

local lp = OrderedMap()
options.O['lantm'] = lp
lp.key_dim = {default=2,
    comment='dimension of the key space. If model = slantm, this is set automatically to 3.'}
lp.mem_width = {default=20,
    comment='size of individual memories'}
lp.num_memory_modules = {default=1,
    comment='number of memory modules to external memory models'}
lp.weight_method = {default='InvNormalize',
    comment='method for computing weightings for memory reads. InvNormalize | SoftMax'}
lp.num_read = {default=1,
    comment='number of read heads per memory'}
lp.init_mem_length = {default=1,
    comment='length of initial memory module. Must be > 0'}
lp.squash_mem = {default=true,
    comment='whether to squash memory vectors with Tanh()'}
lp.bound_act = {default=false,
    comment='bound the norm of actions. false | true. ' ..
            'For LANTM, bounds the translate to have norm at most 1. ' ..
            'For SLANTM, bounds the angle of rotation to a learnable parameter.'}
lp.bias_gate_write = {default=-10,
    comment='additive bias to presigmoid mix coefficient for write key. ' ..
            'The more negative this is, the higher the influence of the previous write key'}
lp.bias_gate_read = {default=0,
    comment='additive bias to presigmoid mix coefficient for read key. ' ..
            'The more negative this is, the higher the influence of the previous read key'}
lp.bias_gate_act_write = {default=-10,
    comment='additive bias to presigmoid mix coefficient for write lie action. ' ..
            'The more negative this is, the higher the influence of ' ..
            'the previous write lie action. ' ..
            'So, if bias_gate_write is also very negative, ' ..
            'then the writes will stay in a straight line'}
lp.bias_gate_act_read = {default=0,
    comment='additive bias to presigmoid mix coefficient for read lie action. ' ..
            'The more negative this is, the higher the influence of ' ..
            'the previous read lie action. ' ..
            'So, if bias_gate_read is also very negative, ' ..
            'then the reads will stay in a straight line'}
lp.bias_init_mem_strength = {default=-10,
    comment='the more negative this is, the closer the initial memory strengths ' ..
            'for learnable initial states is to zero'}
lp.act_interp = {default=false,
    comment='interpolation of candidate action with the action of last time point'}

-- if sumpow = 2 and invnorm = 1/2, then this is invnorm on the 2-norm
-- if sumpow = 2 and invnorm = 1, then this is invnorm on squared euclidean dist
lp.sumpow = {default=2,
    comment='power on the differences btw coordinates' ..
            'used to compute the read weights'}
lp.invnorm_pow = {default=1,
    comment='power for InvNorm'}
lp.epsilon_read_weight = {default=1e-9,
    comment='the smaller this is, the sharper the read'}

--[[ lantm only ]]--
lp.bound_act_exp = {default=2,
    comment='the higher this is, the faster the norm of translate vector saturates. ' ..
            'For lantm.'}
--[[ slantm only ]]--
lp.angle_bound_bias = {default=-5,
    comment='pi * sigmoid(____) would be the initial maximal angle or rotation. ' ..
            'Only applies when bound_act=="learnable"'}

--[[ ram only ]]--
lp.tape = {default=false,
    comment='adding head forward/backward movement to ram model'}
lp.wt_sharpening = {default=false,
    comment='for ram, sharpens the read weights'}

local tp = OrderedMap()
options.O['train'] = tp
tp.batch_size = {default=32,
    comment='size of minibatch'}
tp.iter_size = {default=10,
    comment='size of each iter in terms of forward/backward runs.'}
tp.optimizer = {default='rmsprop',
    comment='any optimizer from the optim package. Ex: rmsprop | adam | adagrad | sgd'}
tp.learning_rate = {default=2e-2,
    comment='learning rate for the optimizer'}
tp.dropout = {default=0.0,
    comment='dropout probability'}
tp.decay_rate = {default=0.99,
    comment='decay for learning rate'}
tp.decay_delay = {default=2,
    comment='decay starts at ___ + 1 iter'}
tp.max_grad_norm = {default=32,
    comment='constrain norm of gradient to be less than this by rescaling'}
tp.max_iter = {default=1e10,
    comment='experiment stops when this iter is reached'}
tp.seed = {default=1,
    comment='random number generator seed'}
tp.valid_interval = {default=50,
    comment='do validation every ___ iters and modify experiment state based on result.'}
tp.valid_after = {default=0,
    comment='do validation only after ___ iters'}
tp.reset_train_interval = {default=50,
    comment='reset train dataset pointer to the beginning every ____ iters.'}

local task = OrderedMap()
options.O['task'] = task
task.dataset = {default='data/copy/copy-32-128-1000_2_64-100_65_128/'}
task.blank_input_during_response = {default=true,
    comment='feed only blank inputs to the model during answer phase'}
task.no_write_during_response = {default=true,
    comment='whether to freeze the memory right before prediction time'}

local report = OrderedMap()
options.O['report'] = report
report.progress_report_interval = {default=2,
    comment='print progress report every ___ iters'}
report.run_vis = {default=false,
    comment='if true, all plot visualization comes from running the model on the first validation batch.'}

local misc = OrderedMap()
options.O['misc'] = misc
misc.gpuid = {default=-1,
    comment='which gpu to use. -1 = use CPU'}
misc.opencl = {default=0,
    comment='use OpenCL (instead of CUDA). Warning: not tested'}
misc.expname = {default='model',
    comment='file name of model checkpoint'}
misc.use_checkpoint = {default='',
    comment='checkpoint name to load from. Not used if string is empty'}
misc.checkpoint_load_mode = {default='model_params',
    comment='how to load checkpoint. ' ..
    'Only effective when a checkpoint is given through "-use_checkpoint". ' ..
    '"complete" | "model_params"'}
misc.checkpoint_dir = {default='checkpoints',
    comment='directory to save checkpoints in'}
misc.checkpoint_freq = {default=10000000,
    comment='checkpoint every ___ iter. Should be a multiple of `valid_interval`'}
misc.exp_id = {default=false,
    comment='experiment ID. If false (default), then it is automatically generated.'}

function options.get_opt(arg, quiet)
    arg = arg or {}
    if type(arg) == 'string' then
        local cjson = require 'cjson'
        arg = Map(cjson.decode(arg))
    end
    assert(types.is_type(arg, Map) or types.is_type(arg, OrderedMap),
        'arg should be Map, OrderedMap, or a json string')
    local opt = OrderedMap()
    for category, opts in options.O:iter() do
        for name, vals in opts:iter() do
            if arg[name] ~= nil then
                opt[name] = arg[name]
            else
                opt[name] = vals.default
            end
        end
    end
    assert(Set(arg:keys()) < Set(opt:keys()),
        'unknown opts: ' .. tostring(Set(arg:keys()) - Set(opt:keys())))
    if opt.model == 'lstm' then
        opt.num_memory_modules = 0
        print('because model is lstm, num_memory_modules is set to 0')
    end
    if opt.model == 'slantm' then
        opt.key_dim = 3
        print('because model is slantm, key_dim is set to 3')
    end
    if opt.model == 'ram' and opt.tape and opt.init_mem_length < 2 then
        opt.init_mem_length = 2
        print('because using ram/tape and init_mem_length < 2, ' ..
            'init_mem_length is set to 2')
    end
    if not opt.exp_id then
        opt.exp_id = torch.round(sys.clock()) .. string.random(16, '%l')
    end
    opt.githash = utils.get_current_git_hash()
    opt.gitbranch = utils.get_current_git_branch()

    if not quiet then
        print(opt)
    end
    return opt
end

return options
