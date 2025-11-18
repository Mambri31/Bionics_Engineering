%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This matlab script measures the moments of a distirbution: mran value, variance, skewness and kurtosis 
% from N independent realizations of the random variable. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N=10^5;   % Numero di prove dell'esperimento (numero di realizzazioni osservate)
etax=1;   % valor medio della v.a. Gaussiana
varx=1;   % varianza della v.a. Gaussiana
etay=1;   % valor medio della v.a. Exp monolatera

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X=sqrt(varx)*randn(1,N)+etax; % Generazione di N campioni di una v.a. Gaussiana
U=rand(N,1);
Y=-log(U)./etay;              % Generazione di N campioni di una v.a. Exp. monolatera 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h=waitbar(0,'Please wait...');
Step=10;         % Le stime si calcolano per un numero di dati che varia a passi di "Step"
for k=1:N/Step   % Si calcolano N/Step stime invece di N stime (si riducono cosi i tempi di simulazione)
    waitbar(k/(N/Step),h); 
    kstep=k*Step;  % Le stime si calcolano per un numero di dati "kstep" che varia a passi di "Step"
    n(k)=kstep;    % Vettore del numero dei dati in corrispeondenza del quale si calcolano le stime
    m1x(k)=mean(X(1:kstep));                    % Stima del valor medio di X
    m1y(k)=mean(Y(1:kstep));                    % Stima del valor medio di Y
    varx(k)=mean((X(1:kstep)-m1x(k)).^2);       % Stima della varianza di X
    vary(k)=mean((Y(1:kstep)-m1y(k)).^2);       % Stima della varianza di Y
    mu3x(k)=mean((X(1:kstep)-m1x(k)).^3);       % Stima del momento centrale di ordine 3 di X
    mu3y(k)=mean((Y(1:kstep)-m1y(k)).^3);       % Stima del momento centrale di ordine 3 di Y
    skewx(k)=mu3x(k)./(sqrt(varx(k))^3);        % Stima della Skewness di X
    skewy(k)=mu3y(k)./(sqrt(vary(k))^3);        % Stima della Skewness di Y
    mu4x(k)=mean((X(1:kstep)-m1x(k)).^4);       % Stima del momento centrale di ordine 4 di X
    mu4y(k)=mean((Y(1:kstep)-m1y(k)).^4);       % Stima del momento centrale di ordine 4 di Y
    kurtx(k)=mu4x(k)./(varx(k)^2)-3;            % Stima della Kurtosis di X
    kurty(k)=mu4y(k)./(vary(k)^2)-3;            % Stima della Kurtosis di Y
end
close(h)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure (1);
semilogx(n,m1x,'blue', 'LineWidth',2);grid on;
hold on
semilogx(n,varx,'red', 'LineWidth',2);
semilogx(n,skewx,'magenta', 'LineWidth',2);
semilogx(n,kurtx,'green', 'LineWidth',2);
legend('Sample Mean','Sample Variance','Sample Skewness','Sample Kurtosis');
title('Sample Estimates of the moments: Gaussian r.v.');xlabel('Sample size (N)');ylabel('Sample Estimates')

figure (2);
semilogx(n,m1y,'blue', 'LineWidth',2);grid on;
hold on
semilogx(n,vary,'red', 'LineWidth',2);
semilogx(n,skewy,'magenta', 'LineWidth',2);
semilogx(n,kurty,'green', 'LineWidth',2);
legend('Sample Mean','Sample Variance','Sample Skewness','Sample Kurtosis');
title('Sample Estimates of the moments: Exponential r.v.');xlabel('Sample size (N)');ylabel('Sample Estimates')

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
