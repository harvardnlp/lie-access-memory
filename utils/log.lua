require 'paths'
local common = paths.dofile('common.lua')
local log = {}

local function _optimal(map, comparison, kw)
    kw = kw or {}
    local inf2str = kw.inf2str or true
    local key, best = tablex.sortv(map, comparison)()
    if best and type(best) == 'number' and not common.isfinite(best) then
        best = tostring(best)
    end
    return key, best
end

function log.captureresults(kwargs)
    kwargs = kwargs or {}
    local results = {}
    local comment = kwargs.comment or ''
    local keys2str = kwargs.keys2str == nil and true or kwargs.keys2str
    local results_success, results_err = pcall(
        function()
            if not success then
                results.error = common.escapestr(tostring(err))
            else
                results.error = ''
            end
            results.iter = __xstate.iter
            results.optim_hist = __optim_hist
            results.modelsize = __params:size()[1]

            results.best_valid_coarse_acc_iter, results.best_valid_coarse_acc =
                _optimal(__xstate.valid_coarse_acc, operator.gt)
            results.best_valid_fine_acc_iter, results.best_valid_fine_acc =
                _optimal(__xstate.valid_fine_acc, operator.gt)
            results.best_valid_loss_iter, results.best_valid_loss =
                _optimal(__xstate.valid_losses, operator.lt)
            results.best_train_coarse_acc_iter, results.best_train_coarse_acc =
                _optimal(__xstate.train_coarse_acc, operator.gt)
            results.best_train_fine_acc_iter, results.best_train_fine_acc =
                _optimal(__xstate.train_fine_acc, operator.gt)
            results.best_train_loss_iter, results.best_train_loss =
                _optimal(__xstate.train_losses, operator.lt)
            results.valid_coarse_acc = __xstate.valid_coarse_acc
            results.valid_fine_acc = __xstate.valid_fine_acc
            results.valid_losses = __xstate.valid_losses
            results.train_coarse_acc = __xstate.train_coarse_acc
            results.train_fine_acc = __xstate.train_fine_acc
            results.train_losses = __xstate.train_losses


            results.total_time = __total_timer and __total_timer:time().real / 60

            results.comment = comment
        end
    )
    if results_err then
        results.results_err = tostring(results_err)
    else
        results.results_err = ''
    end
    if keys2str then
        results = common.keys2str(results)
    end
    return results
end

return log
