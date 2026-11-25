//! Registration of analyses into a pipeline (design §12.2).

use crate::analysis::{Analysis, AnalysisContext, Summary};
use crate::data::property::PropertyStore;
use crate::error::Result;

pub struct Registry {
    analyses: Vec<Box<dyn Analysis>>,
}

impl Default for Registry {
    fn default() -> Self {
        Registry::new()
    }
}

impl Registry {
    pub fn new() -> Self {
        Registry {
            analyses: Vec::new(),
        }
    }

    pub fn add<A: Analysis + 'static>(&mut self, a: A) {
        self.analyses.push(Box::new(a));
    }

    pub fn add_boxed(&mut self, a: Box<dyn Analysis>) {
        self.analyses.push(a);
    }

    pub fn len(&self) -> usize {
        self.analyses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.analyses.is_empty()
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.analyses.iter().map(|a| a.name()).collect()
    }

    /// Ordered union of all cutoffs (ascending, deduped) — the pipeline builds
    /// cell lists up to the max only.
    pub fn cutoffs(&self) -> Vec<f64> {
        self.cutoffs_for_frame(None)
    }

    /// As [`Registry::cutoffs`], but letting each analysis size its search from
    /// the frame (see [`Analysis::cutoffs_for`]).
    pub fn cutoffs_for_frame(&self, frame: Option<&crate::data::frame::Frame>) -> Vec<f64> {
        let mut c: Vec<f64> = self
            .analyses
            .iter()
            .flat_map(|a| match frame {
                Some(f) => a.cutoffs_for(f),
                None => a.cutoffs(),
            })
            .collect();
        c.sort_by(|a, b| a.total_cmp(b));
        c.dedup();
        c
    }

    pub fn run_all(
        &self,
        ctx: &AnalysisContext,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) -> Result<()> {
        for a in &self.analyses {
            a.run(ctx, out, summary)?;
        }
        Ok(())
    }
}
