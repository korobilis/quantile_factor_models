function [EX] = VBPPCA_own(X,q,prior)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[N, d] = size(X);

if isempty(q)
    q = d-1; 
end

q = 1;

% EM algorithm settings
Threshold = 1.0e-6;
F_new     = -1000;
F_old     = 1;
kappa     = 1;
maxiter   = 1000;

% initialize parameters
Etau = 1;  
EW   = rand(d,q); 
EWW  = eye(q);
Emu  = zeros(d,1);
Ealpha = ones(q,1);

%% Start Variational Bayes iterations
%format bank;
%fprintf('\n')
ELL = [];

a_alpha = 10e-3;
b_alpha = 10e-3;
a_tau   = 10e-3;
b_tau   = 10e-3;
beta    = 10e-3;

while kappa < maxiter && max(abs((F_new - F_old))) > Threshold
    %% compute variational density of factors, X
    SigX = eye(q)/(eye(q)+Etau*EWW); 
    %mX   = Etau*SigX*EW'*(X'-Emu);
    
    for n = 1:N
       mX(n,:) = Etau*SigX*EW'*(X(n,:)-Emu')'; 
    end
    
    % compute expectations
    EX   = mX';
    EXX  = SigX + EX*EX';
    
    %% compute variational density of factors, mu
    SigMu  = (beta+N*Etau)\eye(d);
    %mMu    = Etau*SigMu*sum(X'-EW*EX,2);
    
    mMu = zeros(n,d);
    for n = 1:N
        mMu(n,:) = X(n,:)-(EW*EX(:,n))';
    end
    mMu = Etau*SigMu*sum(mMu)';
    
    Emu   = mMu;
    Emumu = SigMu + Emu*Emu';
    
    %% compute variational density of loadings, w
    SigW = eye(q)/(diag(Ealpha) + Etau*EX*EX');
    
    mW = zeros(d,q);
    for k = 1:d
        aux = ones(N,q);
        for n = 1:N
            aux(n,:) = EX(:,n)*(X(n,k)-Emu(k));
        end
        mW(k,:) = Etau*SigW*sum(aux)';
    end
    
    %mW   = Etau*SigW*EX*(X-repmat(Emu',n,1));
    
    EW   = mW;
    EWW  = SigW + EW'*EW;
    
    %% compute variational density for loading variance, alpha
    a_alpha_til = a_alpha + d/2;
    b_alpha_til  = b_alpha + diag(EWW)/2;
    
    a_tau_til   = a_tau + (N*d)/2;
    
    btt = zeros(N,1);
    for n = 1:N
        btt(n,1) = X(n,:)*X(n,:)'  + Emu'*Emu + trace(EWW*(SigX+EX(:,n)*EX(:,n)')) + 2*Emu'*EW*EX(:,n) + 2*X(n,:)*EW*EX(:,n) - 2*X(n,:)*Emu;
    end    
    b_tau_til = b_tau + 0.5*sum(btt);
      
    kappa = kappa+1;

    Ealpha =  a_alpha_til./b_alpha_til;
    Etau   =  a_tau_til/b_tau_til;
end

EX = EX';
end

