function [output] = armaxels(data, orde)

	% Memisahkan vektor y dan u
	y = data(:, 1);
	u = data(:, 2);

	% Mendapat jumlah data dan banyak parameter
	[sizeRow, sizeColumn] = size(data);
	N = sizeRow;
	m = orde * 3;

	% Membuat matrix yang dibutuhkan
	phiArx = zeros(N, 2);
	thetaArx = zeros(2 , 1);
	phi = zeros(N, m);
	theta = zeros(m, 1);

	% Membentuk matriks phi untuk model ARX orde 1
	for k = 1:N
		if k-1 <= 0, phiArx(k, :) = [0 0];
		else
			phiArx(k, :) = [-1*y(k-1) u(k-1)];
		end
	end

	% Menghitung parameter model ARX, dan prediction error
	thetaArx = inv(phiArx' * phiArx) * phiArx' * y;
	yhatArx = phiArx * thetaArx;
	e = y - yhatArx;

	% Membuat matrix phi untuk model ARMAX
	for k = 1:N
		for iterOrder = 1:orde
			if (k - iterOrder) <= 0
				phi(k, iterOrder) = 0;
				phi(k, iterOrder + orde) = 0;
				phi(k, iterOrder + 2*orde) = 0;
			else	
				phi(k, iterOrder) = -1 * y(k-iterOrder);
				phi(k, iterOrder + orde) = u(k-iterOrder);
				phi(k, iterOrder + 2*orde) = -1 * e(k-iterOrder);
			end
		end
	end

	% Menghitung parameter model ARMAX dan estimasi
	theta = inv(phi' * phi) * phi' * y;
	yhat = phi * theta;
    
    % Menghitung FOE
    foe = ((N+orde)/(N-orde)) * sum((y - yhat).^2);
   
    % Menghitung Loss Function
    lossFct = (1/N) * sum((y - yhat))
    
     % Menghitung FPE
    fpe = lossFct / (N-orde)
    
    % Menghitung Noise Variance
    noiseVariance = (e' * e) / (N-orde)
    
    % Parameter
    parameter = struct('a', theta(1,1), 'b', theta(2,1), 'c', theta(3,1))
    
    % Indicators
    indicators = struct('foe',foe,'fpe',fpe,'LossFct', lossFct)
    
    % Validation
    val = struct('foe',foe,'fpe',fpe,'LossFct', lossFct)
    
	output = struct('yhat', yhat, 'parameter', parameter, 'indicators', indicators, 'val', val, 'noiseVar', noiseVariance);
end