R = 580;
C = 890;
L = 24;


coded_mask = double(rand(R,C) >= 0.5);
figure; imagesc(coded_mask); colorbar;

save('coded_mask.mat', 'coded_mask');
