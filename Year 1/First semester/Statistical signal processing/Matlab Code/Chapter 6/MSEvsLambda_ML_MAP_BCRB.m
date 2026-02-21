%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola il Mean Square Error (MSE) degli stimatori
% ML e MAP del segnale costante S aleatorio, distribuito come una v.a di
% Laplace, al variare del parametro Lambda
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M=10^4;                         % Numero di prove Monte Carlo per ogni SNR
N=10;                           % Numero di campioni osservati per ogni stima
varW=1;                        % Varianza del rumore
Lambda=[0.01:0.05:10];          % Vettore dei valori del parametro della Laplace

SNR=2*(Lambda.^2)./varW;        % SNR
SNRdB=10*log10(SNR);            % SNR in dB
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h=waitbar(0,'Please wait...');
for kk=1:length(Lambda)
    waitbar(kk/length(Lambda),h); 
    S=(exprnd(Lambda(kk),M,1)).*(sign(rand(M,1)-0.5));  % Vettore delle M realizzazioni del segnale S
    for i=1:M
        W=sqrt(varW)*randn(N,1);    % generazione del vettore Nx1 di rumore
        Y=S(i)+W;                   % generazione del vettore Nx1 dei dati osservati 
        SML(i)=mean(Y);
        SMAP(i)=(SML(i)-varW./(N*Lambda(kk)))*stepfun(abs(SML(i)),varW./(N*Lambda(kk)));
    end
    ErrML=S-SML';                         % Errore nella stima ML di S
    ErrMAP=S-SMAP';                       % Errore nella stima MAP di S
    MSE_ML(kk)=mean(ErrML.^2);            % MSE nella stima ML di S
    MSE_MAP(kk)=mean(ErrMAP.^2);          % MSE nella stima MAP di S
end
close(h)

BCRB=Lambda.^2./(1+N*SNR/2);

figure (1);
plot(Lambda,10*log10(MSE_ML),'LineWidth',2);grid on;
hold on
plot(Lambda,10*log10(MSE_MAP),'g','LineWidth',2);
plot(Lambda,10*log10(BCRB),'r','LineWidth',2);
axis([0 10 -40 20])
legend('ML estimator','MAP estimator', 'BCRB(s)');
title('MSE of ML and MAP estimates vs the BCRB');xlabel('\lambda');ylabel('MSE [dB]')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 