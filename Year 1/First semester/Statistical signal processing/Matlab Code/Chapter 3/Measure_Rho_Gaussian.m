%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab genera M realizzazioni di due v.a. congiuntamente Gaussiane X ed Y
% e calcola valor medio, varianza, e coefficiente di correlazione di X ed Y
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

M=10^5;
etaX=1;
etaY=2;
varX=3;
varY=4;
rhoXY=0.5;

step=10;  % Passo a cui si calcola e si grafica la stima dei parametri

mz=[etaX etaY]';
Cz=[varX rhoXY*sqrt(varX*varY); rhoXY*sqrt(varX*varY) varY];

% Generazione di vettori correlati mediante metodo di Cholesky

b=mz;            % b è un vettore 2X1
A=(chol(Cz))';   % A è una matrice 2x2

w=randn(2,M);         % genera M coppie di v.a. Gaussiane standard indipendenti
z=A*w+repmat(b,1,M);  % genera M coppie di v.a. Gaussiane correlate con ddp data, z è una matrice 2XM
z=z';                 % z è una matrice MX2

for k=1:M/step,  % stime calcolate per numero di campioni variabile a passi di "step"
    ncamp=k*step;
    etaXhat(k)=mean(z(1:ncamp,1));  
    etaYhat(k)=mean(z(1:ncamp,2));
    varXhat(k)=var(z(1:ncamp,1));  
    varYhat(k)=var(z(1:ncamp,2));
    covXY=mean((z(1:ncamp,1)-etaXhat(k)).*(z(1:ncamp,2)-etaYhat(k)));
    rhoXYhat(k)=covXY/sqrt( varXhat(k)*varYhat(k));
end

Ncamp=step*[1:M/step]';                          
figure(1)
semilogx(Ncamp,etaXhat,'r',Ncamp,etaYhat,'g',Ncamp,varXhat,'m',Ncamp,varYhat,'k',Ncamp,rhoXYhat,'b', 'Linewidth',2)
grid on;
xlabel('Sample Size (N)');
ylabel('Sample Estimates of moments of X and Y')
title('Sample Estimates of moments: X and Y jointly Gaussian')
legend('Sample Mean X','Sample Mean Y','Sample Variance X','Sample Variance Y','Sample Correlation Coefficient')
axis([step M -1 14]);

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


