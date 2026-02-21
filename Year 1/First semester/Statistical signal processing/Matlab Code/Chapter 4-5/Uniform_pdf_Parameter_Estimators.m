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

N=10;
M=10^6;                      % Numero di prove Monte Carlo per ogni eta
Nbin=200;
a=2;
b=5;

x=a+(b-a)*rand(N,M);
eta=mean(x);
sigma=std(x);
ahat=eta-sqrt(3)*sigma;
bhat=eta+sqrt(3)*sigma;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [count1,a1]=hist(ahat,Nbin);    % esegue i conteggi
    Delta1=a1(2)-a1(1);             % calcola larghezza bin
    ddpahat=count1/Delta1/M;        % istogramma normalizzato
    
    [count2,b1]=hist(bhat,Nbin);    % esegue i conteggi
    Delta2=b1(2)-b1(1);             % calcola larghezza bin
    ddpbhat=count2/Delta2/M;        % istogramma normalizzato
   
    figure (1);
    plot(a1,ddpahat,'g','LineWidth',2)   
    hold on
    grid on
    plot(b1,ddpbhat,'b','LineWidth',2)   
    
   
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N=100;
M=10^6;                      % Numero di prove Monte Carlo per ogni eta
Nbin=200;
a=2;
b=5;

x=a+(b-a)*rand(N,M);
eta=mean(x);
sigma=std(x);
ahat=eta-sqrt(3)*sigma;
bhat=eta+sqrt(3)*sigma;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [count1,a1]=hist(ahat,Nbin);    % esegue i conteggi
    Delta1=a1(2)-a1(1);             % calcola larghezza bin
    ddpahat=count1/Delta1/M;        % istogramma normalizzato
    
    [count2,b1]=hist(bhat,Nbin);    % esegue i conteggi
    Delta2=b1(2)-b1(1);             % calcola larghezza bin
    ddpbhat=count2/Delta2/M;        % istogramma normalizzato
   
    figure (1);
    plot(a1,ddpahat,'g-.','LineWidth',2)   
    plot(b1,ddpbhat,'b-.','LineWidth',2)   
    legend('pdf est. a: N=10','pdf est. b: N=10', 'pdf est. a: N=100','pdf est. b: N=100');
    title('Probability density function of the two estimators');
    xlabel('a,b');ylabel('pdf')
    axis([0 8 0 4])
    
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