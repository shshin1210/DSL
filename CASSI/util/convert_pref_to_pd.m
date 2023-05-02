function img_pd = convert_pref_to_pd(img_ref, p_ref)

[N,M,nCh] = size(img_ref);
[n_pd_y,n_pd_x,~] = size(p_ref);

img_pd = zeros(n_pd_y, n_pd_x, nCh);

valid_ind = logical((1 <= p_ref(:,:,1)) .* (p_ref(:,:,1) <= M) .* (1 <= p_ref(:,:,2)) .* (p_ref(:,:,2) <= N));
valid_ind = reshape(valid_ind, [], 1);
p_ref = reshape(p_ref, [], 2);
p_ref = round(p_ref);

nind = sub2ind([N, M] , (p_ref(valid_ind,2)), (p_ref(valid_ind,1)));

img_ref = reshape(img_ref, [], nCh);
img_pd = reshape(img_pd, [], nCh);
for i = 1:nCh
    img_pd(valid_ind, i) = img_ref(uint32( nind ), i);
end

img_pd = reshape(img_pd, n_pd_y, n_pd_x, nCh);
end