class FixedLearnRateScheduler(object):

  def __init__(self, sess, model, base_lr, lr_decay_steps, lr_list=None):
    self.model = model
    self.sess = sess
    self.lr = base_lr
    self.lr_list = lr_list
    self.lr_decay_steps = lr_decay_steps
    self.model.assign_lr(self.sess, self.lr)

  def step(self, niter):
    if len(self.lr_decay_steps) > 0:
      if (niter + 1) == self.lr_decay_steps[0]:
        if self.lr_list is not None:
          self.lr = self.lr_list[0]
        else:
          self.lr *= 0.1  ## Divide 10 by default!!!
        self.model.assign_lr(self.sess, self.lr)
        self.lr_decay_steps.pop(0)
        log.warning("LR decay steps {}".format(self.lr_decay_steps))
        if self.lr_list is not None:
          self.lr_list.pop(0)
      elif (niter + 1) > self.lr_decay_steps[0]:
        ls = self.lr_decay_steps
        while len(ls) > 0 and (niter + 1) > ls[0]:
          ls.pop(0)
          log.warning("LR decay steps {}".format(self.lr_decay_steps))
          if self.lr_list is not None:
            self.lr = self.lr_list.pop(0)
          else:
            self.lr *= 0.1
        self.model.assign_lr(self.sess, self.lr)