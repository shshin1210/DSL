function val = nonneg(v)

% I(x >= 0)
% Evaluate the function on v (ignoring parameters).

if min(v(:)) > -1e-3
    val = 0;
else
    val = inf;
end

end