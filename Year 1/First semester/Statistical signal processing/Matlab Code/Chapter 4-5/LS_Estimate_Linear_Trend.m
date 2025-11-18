%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo programma implementa la stima di un trend lineare dei dati
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

A=2;
B=1;
N=101;
varW=50;

k=[0:N-1]';
W=sqrt(varW)*randn(N,1);
Y=A+B*k+W;

h1=ones(N,1);
h2=[0:N-1]';
H=[h1 h2];

% Stima di A e B

Theta=pinv(H)*Y;
Ahat=Theta(1);
Bhat=Theta(2);
phat=Ahat+Bhat*k;
p=A+B*k;

figure(1)
stem(k,Y,'b','Linewidth', 2);
grid on; hold on
plot(k,p,'r--','Linewidth', 2);
plot(k,phat,'g','Linewidth', 2);
legend('Observed sampled data X[n]', 'True line p(t)', 'Estmated line');
title('Estimate of a linear trend in the data');
xlabel('Continuous-Time [t]');ylabel('X[n], p(t), Estimate of p(t)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
