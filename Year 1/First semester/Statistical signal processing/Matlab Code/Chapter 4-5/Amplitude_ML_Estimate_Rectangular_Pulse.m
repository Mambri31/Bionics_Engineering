%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Stima dell'Ampiezza di un impulso rettangolare in banda base presente in [0,T)
% Nel tempo-discreto l'impulso è presente dall'istante N+1 all'istante 2N.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
%
%
A=10;
M=300;
N=100;
b=ones(N,1);
Es=A^2;      %  SNR=A^2/varW rapporto segnale rumore sul singolo campione
SNRdb=10;
varW=Es*10^(-SNRdb/10);
W=sqrt(varW)*randn(M,1);
s=zeros(M,1);
s(N+1:2*N)=A*b;
X=s+W;
Z=filter(b/sqrt(N),1,X);
Zs=filter(b/sqrt(N),1,s);
Astimato=Z(N+1)/sqrt(N);
%
% Figura di una realizzazione di X(t)
%
figure(1)
plot(-N:M-1-N,X,'b','Linewidth', 2);grid on;
hold on
plot(-N:M-1-N,s,'r--','Linewidth', 2);grid on;
legend('X_{B}(t)','A*s(t)');
title('Observed signal: Rectangular pulse in AWGN')
xlabel('Continuous-time (t)')
ylabel('X_{B}(t), A*s(t)')

% Figura di una realizzazione del segnale all'uscita del filtro adattato

figure(2)
plot(-N:M-1-N,Z,'b','Linewidth', 2);grid on;
hold on
plot(-N:M-1-N,Zs,'r--','Linewidth', 2);grid on;
legend('Z(t)','Z_s(t)');
title('Matched Filter output: Rectangular pulse in AWGN')
xlabel('Continuous-time (t)')
ylabel('Z(t), Z_s(t)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%