function [Weight,Var]=Complex_MMV_SAVE_ReWeLaplace(PHI,X)
%% This is a function for SAVE SBL algorithm with MMV
%% Input:
%  PHI: the dictionary;
%  X  : the MMV observation;
%% Output:
%  Weight: the recovered source signal;
%  Var   : the variance of recovered source signal;
%% Author:
%  Zonglong Bai, baizonglong@gmail.com;
%  Liming Shi,   ls@create.aau.dk;
%  Last modified by Jan.26,2020;
%% Reference:
%  The technical report.

if nargin<2
    error('Not enough inputs!');
end

%% Initialization
a = 1e-3;
b = 1e-3;
c = 1e-3;
d = 1e-3;

M = size(PHI,1);
N = size(PHI,2);
L = size(X,2);

rho   = a/b;
gamma = c/d*ones(N,1);

lambda    = 6./gamma;
lambdaInv = 1./lambda;

PHP     = PHI'*PHI;
diagPHP = diag(PHP);

mu_gDist     = zeros(N,L);
sigma2_gDist = 1./(L*rho*diagPHP+lambdaInv);
g            = zeros(N,L);

IterMax = 1000;
Iter    = 1;
errMin  = 1e-6;
err     = 1;

dataTmp_g = zeros(M,L);
%PHIExt    = repmat(PHI,[1,1,L]);
%PHI_MMV   = PHI*ones(N,L);

while Iter<IterMax && err>errMin
    g_old = g;
    for i = 1:N
        for j=1:L
        dataTmp_X=squeeze(X(:,j));
        % Update g_i~Gaussian distribution
        sigma2_gDist(i) = 1./(rho*diagPHP(i)+lambdaInv(i));
        dataTmp_g(:,j)    = dataTmp_g(:,j)-PHI(:,i).*mu_gDist(i,j);
        mu_gDist(i,j)     = sigma2_gDist(i)*rho*PHI(:,i)'*(dataTmp_X-dataTmp_g(:,j));
        dataTmp_g(:,j)    = dataTmp_g(:,j)+PHI(:,i).*mu_gDist(i,j);
        g(i,j)            = mu_gDist(i,j);
        end
        % Update lambda_i~generalizaed Gaussian distribution
        a_lambdaDist = 0.5*gamma(i);
        b_lambdaDist = 2*(mu_gDist(i,:)*mu_gDist(i,:)'+L*sigma2_gDist(i));
        lambda(i)    = sqrt(b_lambdaDist)/sqrt(a_lambdaDist)*(1+1/sqrt(a_lambdaDist*b_lambdaDist));
        lambdaInv(i) = sqrt(a_lambdaDist)/sqrt(b_lambdaDist)*(1+1/sqrt(a_lambdaDist*b_lambdaDist))-1/b_lambdaDist;
        % Update gamma_i~Gamma distribution
        gamma(i) = (c+L+0.5)./(lambda(i)/4+d);
    end
    % Update rho~Gamma distribution
    dataTmp_rho   = 0;
    dataTmp_sigma = 0;
    for j=1:L
        dataTmp_rho   = dataTmp_rho+norm(X(:,j)-PHI*mu_gDist(:,j))^2;
        dataTmp_sigma = dataTmp_sigma+sum(sigma2_gDist.*diagPHP(:));
    end
    rho         = (L*M+a)./(dataTmp_rho+dataTmp_sigma+b);
    Iter        = Iter+1;
    err         = norm(g-g_old);
end
Weight = g;
Var    = sigma2_gDist;  

end