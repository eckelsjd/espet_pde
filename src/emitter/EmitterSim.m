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
        function self = EmitterSim(d, rc, alpha, h, ra, te, V0, ms)
            self.d = d; self.rc = rc; self.alpha = alpha; self.h = h;
            self.ra = ra; self.te = te; self.V0 = V0; self.ms = ms;
        end

        function solve(self, geo, v_edges, z_edges)
            permittivity_air = 1;
            growth_rate = 1.05;
            refine_factor = 8;
            self.emagmodel = createpde('electromagnetic', 'electrostatic-axisymmetric');
            geometryFromEdges(self.emagmodel, geo);
            self.emagmodel.VacuumPermittivity = 8.8541878128E-12;
            electromagneticBC(self.emagmodel, 'Voltage', self.V0, 'Edge', v_edges);
            [max_vals, ~] = max(self.emagmodel.Geometry.Vertices);
            ymax = max_vals(2);
            right_bc = @(location,~) (self.V0/(ymax))*location.y;
            electromagneticBC(self.emagmodel, 'Voltage', right_bc, 'Edge', v_edges(end)+1);
            electromagneticBC(self.emagmodel, 'Voltage', 0, 'Edge', z_edges);
            electromagneticSource(self.emagmodel, 'ChargeDensity', 0, 'Face', 1);
            electromagneticProperties(self.emagmodel, 'RelativePermittivity', permittivity_air, 'Face', 1);
            generateMesh(self.emagmodel, 'Hmax', self.ms*refine_factor, 'Hgrad',...
                growth_rate, 'Hedge', {[self.e_edges(1)], self.ms, ...
                [self.e_edges(2:end)], self.ms});
            self.emagresults = solve(self.emagmodel);
        end
    end
end