'use strict';

MCMC.registerAlgorithm('HamiltonianMC', {

  description: 'Hamiltonian Monte Carlo',

  about: () => {
    window.open('https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo');
  },

  init: (self) => {
    self.leapfrogSteps = 37;
    self.dt = 0.1;
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, 'leapfrogSteps', 5, 100).step(1).name('Leapfrog Steps');
    folder.add(self, 'dt', 0.05, 0.5).step(0.025).name('Leapfrog &Delta;t');
    folder.open();
  },

  step: (self, visualizer) => {

    const q0 = self.chain.last();
    const p0 = MultivariateNormal.getSample(self.dim);

    // use leapfrog integration to find proposal
    var q = q0.copy();
    var p = p0.copy();
    var trajectory = [q.copy()];
    const momenta = [p.copy()];
    let L = 0
    for (let i = 0; i < self.leapfrogSteps; i++) {
      var q1, p1, pre_q, pre_p;
      pre_q, pre_p = q.copy(), p.copy();
      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      q.increment(p.scale(self.dt/0.5));
      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      trajectory.push(q.copy());
      momenta.push(p.copy());
      L++;
      if (i == 0) {
        q1, p1 = q.copy(), p.copy()
      } else {
        if ((p.dot(p0))||(i+1 == self.leapfrogSteps)) {
          let len = trajectory.length;
          // const randIdx = Math.floor(Math.random()*(len-1)) + 1;
          // q = trajectory[randIdx];
          // p = momenta[randIdx];

          break
        }
      }
    }
    let len = trajectory.length;
    var _q0 = trajectory[len-1].copy();
    var _p0 = momenta[len-1].copy().scale(-1);
    var _q = _q0.copy();
    var _p = _p0.copy();
    for (let i = 0; i < self.leapfrogSteps; i++) {
      var _q1, _p1, _pre_q, _pre_p;
      _pre_q, _pre_p = _q.copy(), _p.copy();
      _p.increment(self.gradLogDensity(_q).scale(self.dt / 2));
      _q.increment(_p.scale(self.dt/0.5));
      _p.increment(self.gradLogDensity(_q).scale(self.dt / 2));
      trajectory.push(_q.copy());
      momenta.push(_p.copy());
      if (i == 0) {
        _q1, _p1 = _q.copy(), _p.copy()
      } else {
        if ((_p.dot(_p0) <= 0)||(i+1 == self.leapfrogSteps)) {
          break
        }
      }
    }
    console.log("L q0 q1",L, q0, _q)

    // add integrated trajectory to visualizer animation queue
    visualizer.queue.push({ 
      type: 'proposal', 
      proposal: q, 
      trajectory: trajectory, 
      initialMomentum: p0 
    });

    // calculate acceptance ratio
    const H0 = -self.logDensity(q0) + p0.norm2() / 2;
    const H = -self.logDensity(q) + p.norm2() / 2;
    const logAcceptRatio = -H + H0;

    // accept or reject proposal
    if (Math.random() < Math.exp(logAcceptRatio)) {
      self.chain.push(q.copy());
      visualizer.queue.push({ type: 'accept', proposal: q });
    } else {
      self.chain.push(q0.copy());
      visualizer.queue.push({ type: 'reject', proposal: q });
    }
  }

});