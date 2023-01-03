clc; close all; clear all;
addpath('E:\VE\Malih\PHD\math\final\example');
figure
subplot(2,1,1)
imgage=imread('Malih256.jpg');
imshow(imgage) 
title('Original Image')

hold on
subplot(2,1,2)
% Covert 3D Array of Image to 2D Grayscale Imag
img2d = rgb2gray(imgage);
imshow(img2d)

title('2D Array Grayscale Image')

img = double(img2d);

% rgb2ind()

% read image and add the mask

img = double(img2d)/256;
%  img = img(141:140+256, 51:50+256);
msk = zeros(size(img));
msk(65:192,65:192) = imresize(imread('text.png'), 0.5);
img_corrupted = img;
img_corrupted(msk > 0) = nan;
% create a matrix X from overlapping patches
ws = 16; % window size
no_patches = size(img, 1) / ws;
X = zeros(no_patches^2, ws^2);
k = 1;
for i = (1:no_patches*2-1)
    for j = (1:no_patches*2-1)
        r1 = 1+(i-1)*ws/2:(i+1)*ws/2;
        r2 = 1+(j-1)*ws/2:(j+1)*ws/2;
        patch = img_corrupted(r1, r2);
        X(k,:) = patch(:);
        k = k + 1;
    end
end

% apply Robust PCA
lambda = 0.02; % close to the default one, but works better
tic
[L, S] = RobustPCA(X, lambda, 1.0, 1e-5);
toc

% reconstruct the image from the overlapping patches in matrix L
img_reconstructed = zeros(size(img));
img_noise = zeros(size(img));
k = 1;
for i = (1:no_patches*2-1)
    for j = (1:no_patches*2-1)
        % average patches to get the image back from L and S
        % todo: in the borders less than 4 patches are averaged
        patch = reshape(L(k,:), ws, ws);
        r1 = 1+(i-1)*ws/2:(i+1)*ws/2;
        r2 = 1+(j-1)*ws/2:(j+1)*ws/2;
        img_reconstructed(r1, r2) = img_reconstructed(r1, r2) + 0.25*patch;
        patch = reshape(S(k,:), ws, ws);
        img_noise(r1, r2) = img_noise(r1, r2) + 0.25*patch;
        k = k + 1;
    end
end
img_final = img_reconstructed;
img_final(~isnan(img_corrupted)) = img_corrupted(~isnan(img_corrupted));

% show the results
figure;
subplot(2,2,1), imshow(img_corrupted), title('Corrupted image')
subplot(2,2,2), imshow(img_final), title('Recovered image')
subplot(2,2,3), imshow(img_reconstructed), title('Recovered low-rank L')
subplot(2,2,4), imshow(img_noise), title('Recovered sparse S')
%% comparision L+S should be original image

figure
L=img_reconstructed;
S=img_noise;
subplot(1,2,1)
imshow(img_corrupted)
title('Original Image with text')
subplot(1,2,2)
imshow(L+S)
title('Reconstructed sum L+S')
% fprintf(1, 'ws=%d\tlambda=%f\trank(L)=%d\tcard(S)=%d\terr=%f\n', ...
%        ws, lambda, rank(L), nnz(S), norm(img - img_final, 'fro'));




%% RPCA function

function [L, S] = RobustPCA(X, lambda, mu, tol, max_iter)
    % - X is a data matrix (of the size N x M) to be decomposed
    %   X can also contain NaN's for unobserved values
    % - lambda - regularization parameter, default = 1/sqrt(max(N,M))
    % - mu - the augmented lagrangian parameter, default = 10*lambda
    % - tol - reconstruction error tolerance, default = 1e-6
    % - max_iter - maximum number of iterations, default = 1000

    [M, N] = size(X);
    unobserved = isnan(X);
    X(unobserved) = 0;
    normX = norm(X, 'fro');

    % default arguments
    if nargin < 2
        lambda = 1 / sqrt(max(M,N));
    end
    if nargin < 3
        mu = 10*lambda;
    end
    if nargin < 4
        tol = 1e-6;
    end
    if nargin < 5
        max_iter = 1000;
    end
    
    % initial solution
    L = zeros(M, N);
    S = zeros(M, N);
    Y = zeros(M, N);
    
    for iter = (1:max_iter)
        % ADMM step: update L and S
        L = Do(1/mu, X - S + (1/mu)*Y);
        S = So(lambda/mu, X - L + (1/mu)*Y);
        % and augmented lagrangian multiplier
        Z = X - L - S;
        Z(unobserved) = 0; % skip missing values
        Y = Y + mu*Z;
        
        err = norm(Z, 'fro') / normX;
        if (iter == 1) || (mod(iter, 10) == 0) || (err < tol)
            fprintf(1, 'iter: %04d\terr: %f\trank(L): %d\tcard(S): %d\n', ...
                    iter, err, rank(L), nnz(S(~unobserved)));
        end
        if (err < tol) break; end
    end
end

function r = So(tau, X)
    % shrinkage operator
    r = sign(X) .* max(abs(X) - tau, 0);
end

function r = Do(tau, X)
    % shrinkage operator for singular values
    [U, S, V] = svd(X, 'econ');
    r = U*So(tau, S)*V';
end



