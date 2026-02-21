clear all
close all
clc

sigma_A = 2;

sigma_W = 2;

F_0 = 0.1;

N = 301;

n = linspace(0,10,N);


A = raylrnd(sqrt(sigma_A));

theta = -pi * 2 *pi *randn();

w = randn(1,N)*sigma_W;

f = A* cos(2*pi *F_0*n+theta)+w;

stem (f)








