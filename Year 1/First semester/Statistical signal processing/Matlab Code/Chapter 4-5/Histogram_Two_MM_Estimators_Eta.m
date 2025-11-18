%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola l'istogramma di due diversi stimatori del
% parametro eta di una ddp Exp monolatera ottenuti mediante il metodo dei momenti
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nbin=200;
N=100;
M=10^6;                      % Numero di prove Monte Carlo per ogni eta
eta=1;
W1=randn(M,N);           
W2=randn(M,N);           
X=(W1.^2+W2.^2)/(2*eta);
Eta1=1./mean(X,2);
M1=repmat(Eta1,1,N);
Y=(X-M1).^2;
Eta2=1./sqrt(mean(Y,2));
MSE1=mean((Eta1-eta).^2);
MSE2=mean((Eta2-eta).^2);

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [count1,a1]=hist(Eta1,Nbin);    % esegue i conteggi
    Delta1=a1(2)-a1(1);             % calcola larghezza bin
    ddpEta1=count1/Delta1/M;        % istogramma normalizzato
    
    [count2,a2]=hist(Eta2,Nbin);    % esegue i conteggi
    Delta2=a2(2)-a2(1);             % calcola larghezza bin
    ddpEta2=count2/Delta2/M;        % istogramma normalizzato
   
    figure (2);
    plot(a1,ddpEta1,'g','LineWidth',2)   
    hold on
    grid on
    plot(a2,ddpEta2,'b','LineWidth',2)   
    legend('pdf estimator 1','pdf estimator 2');
    title('Probability density function of the two estimators of \eta');
    xlabel('\eta');ylabel('pdf')
    axis([0 3 0 4])
    
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