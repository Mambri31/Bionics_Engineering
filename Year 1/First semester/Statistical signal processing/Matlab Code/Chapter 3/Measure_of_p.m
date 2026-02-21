%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This Matlab script calculates the frequency of occurrence of the event
% [A_k=1] whose probability is p
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
% close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N=10^5;              % Number of independent trials
n=[1:N]';
uno=ones(1,N);
p=0.1;               % Pr[A_k=1]=p

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A=floor(p+rand(N,1));  % generazione del vettore dei dati binari A


h=waitbar(0,'Please wait...');
for k=1:N
    waitbar(k/N,h); 
    f(k)=sum(A(1:k))/k;         % generazione del vettore della frequenza relativa F
end
close(h)

figure (1);
stem(n(1:50),A(1:50),'blue', 'LineWidth',2);
grid on;
legend('Sequence of binary symbols A_k','Discrete Time (k)');
title('Sequence of binary symbols A_k vs k');xlabel('Discrete Time (k)');ylabel('A_k')
axis([1 50 -1 2])


figure (2);
semilogx(n,f,'blue', 'LineWidth',2);
grid on;
hold on
semilogx(n,p*uno,'red', 'LineWidth',1.5);
legend('Frequency of occurrence F','Probability p');
title('Frequency of occurrence vs N');xlabel('Sample size (N)');ylabel('F_N')
axis([1 N 0 1])

figure (3);
loglog(n,f,'blue', 'LineWidth',2);
grid on;
hold on
loglog(n,p*uno,'red', 'LineWidth',1.5);
legend('Frequency of occurrence F','Probability p');
title('Frequency of occurrence vs N');xlabel('Sample size (N)');ylabel('F_N')
axis([1 N 0 1])

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
