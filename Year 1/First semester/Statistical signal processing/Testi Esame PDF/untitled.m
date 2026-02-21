clear all
close all
clc


gamma = 0:0.1:100;

MSE = 10 .* log(1./(1+10*gamma));

plot(MSE)