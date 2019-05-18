 function [data] = hw_1(nn, b1, b2)
    if nargin < 3       
        nn = 101;
        b1 = 0.06;
        b2 = 1.2;
    end
    m = 0.1;
    data.R = 200e-6;    % m
    data.ro = 100e-6;   % m
    data.w = 4e-6;      % m
    data.A = 0.2;
    data.nn = nn;
    data.ne = data.nn - 1;
    data.rho = 1000.0;  % kg/m^3
    data.K = 1e9;       % Pa (N/m^2)
    data.G = 100e6;     % Pa (N/m^2)
    data.b1 = b1;
    data.b2 = b2;
    data.c = sqrt((data.K+4*data.G/3)/data.rho);
    data.h = data.R/data.ne;
    
    data.mesh.X = linspace(0, data.R, data.nn);
    data.mesh.conn = [1:data.nn-1; 2:data.nn];
    
    data.v = zeros(data.nn,1);
    data.a = zeros(data.nn,1);
    data.d = zeros(data.nn,1);
    
    data.energy.dissipated = 0.0;
    data.rho_current = data.rho;
    data.eps_vol_previous = 0.0;
   
    M = mass_matrix(data);
    
    data.dt = m * data.h/data.c;
    total_time = (data.R)/data.c;
    num_steps = ceil(total_time/data.dt);
    
    history = [];
    for ts = 1:num_steps
        t = (ts-1)*data.dt;
        data.energy.pe = 0.0;
        [f, data] = internal_force(data);
        data.a = f./M;
        if ts > 1
            data.v = data.v + 0.5*data.dt*data.a;
        end
        data.a(1) = 0.0; % Prevent rigid-body motion
        data.v = data.v + 0.5*data.dt*data.a;
        data.d = data.d + data.dt*data.v;
        
        if any(isnan(data.v))
            fprintf('Simulation unstable at step %d.\n', ts);
            break;
        end
        
        data.pressure(1) = 2*data.pressure(1);
        data.pressure(end) = 2*data.pressure(end);
        data.rho_current(1) = 2*data.rho_current(1);
        data.rho_current(end) = 2*data.rho_current(end);
        
        if mod(ts, 100) == 0
            if nargin > 0
                continue;
            end
            data.energy.ke = kinetic_energy(data, M);
            data.energy.total = data.energy.ke + data.energy.pe + ... 
                                data.energy.dissipated;
            history(end+1, :) = [t, data.energy.ke, data.energy.pe, ...
                                 data.energy.dissipated, data.energy.total];

            fprintf('ts ke      pe      W_dis   W_total \n');
            fprintf('%d  %6.5f %6.5f %6.5f %6.5f \n', ...
                   ts, data.energy.ke, data.energy.pe, data.energy.dissipated, ...
                   data.energy.total);                 
            figure(1); clf;
            plot(history(:,1), history(:,[2,3,4,5]),'.-');
            legend('KE', 'PE', 'W^{dissipated}', 'W^{total}');
            pause(0.01);
            
            figure(2); clf;
            plot(data.mesh.X, data.rho_current);
            pause(0.1);
        end
    end  
end

function [M] = mass_matrix(data)
    M = zeros(data.nn,1);
    for c = data.mesh.conn
        Xe = data.mesh.X(c);
        for q = quadrature_1D(2)
            [N, dNdp] = shape(q);
            J0 = Xe*dNdp;   % l_e/2 essentially
            X = Xe*N;
            M(c) = M(c) + N*data.rho*2*pi*X*det(J0)*q(end);           
        end        
    end
    assert(abs(sum(M)-(pi*data.R^2*data.rho)) < 1e-5);
end

function [fint, data] = internal_force(data)
    fint = zeros(data.nn,1);
    data.pressure = zeros(data.nn,1);
    data.rho_current = zeros(data.nn,1);
    for c = data.mesh.conn
        de = data.d(c);
        Xe = data.mesh.X(c);
        ve = data.v(c);
        for q = quadrature_1D(1)
            [N, dNdp] = shape(q);
            J0 = Xe*dNdp;
            dNdX = dNdp/J0;
            X = Xe*N;
            B = [dNdX'; N'/X; [0 0]];
            eps_thermal_vol = data.A*exp(-((X-data.ro)^2)/(data.w^2));
            eps = B*de - (eps_thermal_vol/3)*[1;1;1];
            eps_vol = sum(eps);
            D = B*ve;
            eps_vol_dot = sum(D);
            Q = 0.0;
            if eps_vol_dot < 0.0
                Q = data.b1*data.rho*data.c*data.h*abs(eps_vol_dot) ...
                  + data.b2*data.rho*data.h^2*eps_vol_dot^2;
            end
            p = -data.K*(log(eps_vol+1)/(eps_vol+1)) + Q;
            eps_dev = eps - (eps_vol/3)*[1;1;1];
            T_dev = 2*data.G*eps_dev;
            T = T_dev - p*[1;1;1];
            wt = 2*pi*X*det(J0)*q(end);
            fint(c) = fint(c) - B'*T*wt; 
            data.energy.dissipated = data.energy.dissipated ...
                                   - Q*eps_vol_dot*data.dt*wt;
            data.energy.pe = data.energy.pe + (0.5*data.K*log(1+eps_vol)^2 ...
                           + data.G*(eps_dev'*eps_dev))*wt;
            rho_current = data.rho / (1+eps_vol);
            data.pressure(c) = data.pressure(c) + N*p;
            data.rho_current(c)  = data.rho_current(c) + N*rho_current;
        end
    end
end

function [ke] = kinetic_energy(data, M)
    ke = 0.5*dot(data.v, M.*data.v);
end

function [q] = quadrature_1D(n)
    if n == 1
        q = [0; 2];
    elseif n == 2
        q = [[-1 1]/sqrt(3.0); [1 1]];
    end
end

function [N, dNdp] = shape(q)
    N = 0.5*[1-q(1); 1+q(1)];
    dNdp = 0.5*[-1; 1];
end