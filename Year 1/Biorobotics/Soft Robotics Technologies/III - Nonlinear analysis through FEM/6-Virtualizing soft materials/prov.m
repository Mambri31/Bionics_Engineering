clear all
close all
clc

SS = readtable("tension-uni.xls","Range","A2:B66");

strain = SS{:,1};
stress = SS{:,2};
stretch = 1 + strain;