classdef (Abstract) EmitterSim < handle
    properties
        % Arguments
        d {mustBeNumeric}           % tip-to-extractor distance [m]
        rc {mustBeNumeric}          % radius of curvature of tip [m]
        alpha {mustBeNumeric}       % cone half-angle [rad]
        h {mustBeNumeric}           % height of emitter [m]
        ra {mustBeNumeric}          % radius of emitter aperture [m]
        te {mustBeNumeric}          % thickness of extractor [m]
        V0 {mustBeNumeric}          % Bias voltage [V]
        ms {mustBeNumeric}          % mesh size [m]
        emagmodel                   % electromagnetic PDE model
        emagresults                 % electromagnetic solution results
    end

    properties (Abstract)
        e_edges % emitter edges
    end

    methods (Abstract)
        geo = constructGeometry(self);
    end

    methods
        function self = EmitterSim(d, rc, alpha, h, ra, te, V0)
            self.d = d; self.rc = rc; self.alpha = alpha; self.h = h;
            self.ra = ra; self.te = te; self.V0 = V0;

            % Estimate the mesh size with N discretization at the tip
            % self.ms is (min) mesh size along emitter surface
            N = 25;
            dtheta = ((pi/2)-alpha)/N;
            ds = rc*dtheta;
            self.ms = ds;
        end

        function solve(self, geo, v_edges, z_edges)
            permittivity_air = 1;

            self.emagmodel = createpde('electromagnetic', 'electrostatic-axisymmetric');
            geometryFromEdges(self.emagmodel, geo);

            % Determine max mesh size automatically based on max # of nodes
            growth_rate = 1.05;
            refine_factor = 8;
            ms_max = refine_factor*self.ms;
            Nmax = 15000;          % max nodes 
            [max_vals, ~] = max(self.emagmodel.Geometry.Vertices);
            xmax = max_vals(1); ymax = max_vals(2);
            [min_vals, ~] = min(self.emagmodel.Geometry.Vertices);
            xmin = min_vals(1); ymin = min_vals(2);
            H = abs(ymax - ymin);  % Scene height
            W = abs(xmax - xmin);  % Scene width
            N_est = (H*W)/(ms_max^2);
            if (N_est > Nmax)
                ms_max = sqrt(H*W/Nmax);
            end

            self.emagmodel.VacuumPermittivity = 8.8541878128E-12;
            electromagneticBC(self.emagmodel, 'Voltage', self.V0, 'Edge', v_edges);
            right_bc = @(location,~) (self.V0/(ymax))*location.y;
            electromagneticBC(self.emagmodel, 'Voltage', right_bc, 'Edge', v_edges(end)+1);
            electromagneticBC(self.emagmodel, 'Voltage', 0, 'Edge', z_edges);
            electromagneticSource(self.emagmodel, 'ChargeDensity', 0, 'Face', 1);
            electromagneticProperties(self.emagmodel, 'RelativePermittivity', permittivity_air, 'Face', 1);

            % Regenerate mesh automatically if mesh quality sucks
            bad_mesh = true;
            iter = 1;
            while bad_mesh
                if iter > 10
%                     error('Could not fix mesh :(');
                    throw(MException('EmitterSim:badMesh', 'Poor mesh quality'))
                end
                % THIS BREAKS FOR HYPERBOLOID.m DISCRETIZATION
                generateMesh(self.emagmodel, 'Hmax', ms_max, 'Hgrad',...
                    growth_rate, 'Hedge', {[self.e_edges(1)], self.ms, ...
                    [self.e_edges(2:end)], (self.ms+ms_max)/2});
%                 generateMesh(self.emagmodel, 'Hmax', ms_max, 'Hgrad',...
%                     growth_rate, 'Hedge', {[self.e_edges(1:end)], self.ms});
%                 generateMesh(self.emagmodel, 'Hmax', 2*self.ms, 'Hgrad',...
%                     growth_rate, 'Hedge', {[self.e_edges(1)], self.ms, ...
%                     [self.e_edges(2:end)], self.ms});
                Q = meshQuality(self.emagmodel.Mesh);
                bad_mesh = any(Q<0.5);

                if bad_mesh
                    self.ms = 1.25*self.ms;
                    iter = iter + 1;
                    fprintf('Mesh sucks: take %i\n',iter)
                else
                    if iter>1
                        fprintf('Fixed mesh: ms=%.2E [m]\n',self.ms);
                    end
                end
            end
            
            self.emagresults = solve(self.emagmodel);
        end
    end
end