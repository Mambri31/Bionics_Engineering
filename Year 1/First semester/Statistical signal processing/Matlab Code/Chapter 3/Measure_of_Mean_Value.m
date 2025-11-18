%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This Matlab sript calculates the Sample Mean of N IID realizations of a
% (Gaussian) random variable X
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
% close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N=10^4;              % Numero di prove dell'esperimento
n=[1:N]';

etax=1;  
varx=10;
uno=ones(1,N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x=sqrt(varx)*randn(1,N)+etax; % Si generano N campioni di una v.a. Gaussiana

h=waitbar(0,'Please wait...');
for k=1:N
    waitbar(k/N,h); 
    z(k)=mean(x(1:k));
    boundinf(k)=etax-3*sqrt(varx/k);
    boundsup(k)=etax+3*sqrt(varx/k);
end
close(h)

figure (1);
semilogx(n,z,'blue', 'LineWidth',2);grid on;
hold on
semilogx(n,etax*uno,'red', 'LineWidth',2);
legend('Sample Mean','Statistical Mean');
title('Sample Mean vs N');xlabel('Sample size (N)');ylabel('Sample Mean')
% axis([1 N 0 2])

figure (2);
semilogx(n,z,'blue', 'LineWidth',2);grid on;
hold on
semilogx(n,boundinf,'red -.', 'LineWidth',2);
semilogx(n,boundsup,'red-.', 'LineWidth',2);
legend('Sample Mean','Interval at Prob.=0.997 (k=3)');
title('Sample Mean vs N');xlabel('Sample size (N)');ylabel('Sample Mean')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%
% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

 