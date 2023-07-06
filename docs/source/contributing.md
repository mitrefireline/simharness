# Contributing

Currently, contribution to SimHarness2 is limited to the MITRE Employees currently 
working on the FireLine project. Eventually, SimHarness2 will be deployed to [FireLine's 
public facing GitHub account](https://github.com/mitrefireline) and the process will 
need to be updated accordingly.


## Contributing for MITRE Employees

Contributing to the [MITRE internal GitLab SimHarness codebase](https://gitlab.mitre.org/fireline/reinforcementlearning/simharness2) 
requires a specific workflow process to be followed. This page will explain how to 
follow that process and will provide some tools to make local development easier.

### Overall Process

This process should be followed for every issue that is created for SimHarness.

1. Create a new issue by following [Issue Creation Process](#creating-issues).
    - Make sure the labels include the appropriate [labels](#labels)!
2. Create a new MR from the new issue by following [MR Creation Process](#creating-merge-requests).
3. Develop the MR following the [Workflow Stages Process](#workflow-stages).
    - Make sure to rebase changes from dev following the [Rebase Process](#rebasing-and-merging).
4. Complete a [code review](#code-reviews) for your MR.
    - Get approvals according to [Protect Branches](#protected-branches).
5. Merge the approved branch into `dev` using [Merging Process](#merge-process).
6. Once enough changes to `dev` have been made, merge `dev` into `main`.

### Protected Branches

Protected branches are branches within git that changes cannot be directly pushed to.
This is to ensure that only project reviewed changes can be pushed to project-wide or
public-facing branches in order to limit the pushed mistakes and bugs.

**Protected Branches**
- `main`: Our public-facing branch with only stable, tagged versions of SimHarness.
    - Nobody can push changes directly to this branch.
    - Maintainers only are allowed to merge feature branches into this branch.
    - Two approvals are necessary to merge feature branches into this branch.
- `dev`: Our main project-wide internal development branch.
    - Maintainers only can push directly to this branch for small, immediate changes.
    - Maintainers and Developers can merge feature branches into this branch.
    - One approval is necessary to merge feature branches into this branch.

### Workflow Stages

To maintain order within the development process, SimHarness2 maps each development task
to a specific stage. These workflow stages label each issue created by its position
within the development process. The workflow stages are outlined below:

1. `Open`: These are issues that have been created but are not currently being worked on
in the near future. New ideas for features or non-important changes should be placed
here.
2. `To-Do`: These are issues that are planned to be worked on during the current sprint.
Issues placed within this stage are expected to be completed by the next meeting.
3. `In-Progress`: These are issues that are currently being worked on. An MR should be
open for any issues within this stage. MRs within this stage should be marked `Draft`.
4. `Review`: These are issues that have been completed and are awaiting review by other
members. MRs within this stage should be marked `Resolve`.
5. `Completed`: These are issues that have been reviewed and merged into `dev` before
the end of the sprint. This stage is primarily used to keep track of what has been
completed during a single sprint phase.
6. `Closed`: These are issues that have been reviewed and merged and were part of a
previous sprint.
7. `On-Hold`: This lesser used stage is for issues that started, but got put on-hold. An
issue can be placed on-hold due to blockers or a change in priority, for example.

As you work through the worlflow process, **make sure** to move your issue accordingly.
This helps all members identify what is currently being worked on, what has been
completed, and what _needs_ to be completed by the end of the sprint.

### Rebasing and Merging

Rebasing and merging are very similar processes for combining work from two separate
branches. However, rebasing is typically considered the _cleaner_ option as it will
re-write the commit history to move the feature branch to the end of the target branch
as opposed to incoporating a separate merge commit each time.

For our use case, we want to rebase when adding commits from a lower branch but merge
when adding commits from a higher branch. For example:

Two people are working on `feature1` and `feature2` separately. `feature1` finishes
before `feature2`. As `feature1` is a higher branch than `dev`, we will **merge**
`feature1` into `dev`. Since `dev` has now changed, `feature2` branch needs to be 
updated. As `dev` is a lower branch than `feature2`, we will **rebase** `dev` into 
`feature2`. Once `feature2` is complete, we now **merge** `feature2` into `dev`.

Make sure your feature branches are always up to date with the current state of
`dev`, and make sure to **rebase** these changes each time something gets merged
into `dev`.

### Creating Issues

Issues outline new features, bugs, or updates that document the changes to the overall
codebase. Each issue has a number of labels assigned to it that provide extra hinting
towards what the issue is working on. To create a new issue, follow the instructions
below:

1. Create a new issue.
    - Issue Board: In the appropriate stage list, click the three vertical dots.
    - Issue List: Click the blue `New Issue` button in the top right.
2. Add a short, descriptive title outlining the issue.
3. Add a description outlining in depth what this issue is fixing or adding.
4. Assign the issue to yourself (or somebody else).
5. In labels, select the appropriate labels.
6. Click the blue `Create issue` button.

Issues should be smaller, reasonable issues to avoid large, unwieldy MRs. Smaller issues
are easier to review and detect problems with and result in faster collaboration and 
development overall. Larger issues should invlude subtasks within them to outline
why the process requires a single issue and show progress within the issue.

#### Labels

There are a variety of different labels available to be placed on each label. These
labels help quickly identify what an issue is about and allow for issues to be filtered.
The following label groups are available and should be used for **every** issue created.

**Priority**

- `High`: Requires immediate attention and should be completed before anything else.
- `Medium-High`: Not an immediate priority but is blocking the work of others.
- `Medium`: Not blocking the work of others, but is an important issue.
- `Low`: Not necessary and mainly quality-of-life or nice-to-have additions.

**Project**

- `FireLine Management`
- `JANUS`
- `SimFire`
- `SimHarness`
- `V&V`

**Scoping**

- `Deliverable`: Should be completed by the end of the sprint.
- `Stretch`: Would be nice to have by the end of th sprint, if possible.
- `Next Sprint`: Can be worked on if everything else is completed prior.

**Stage**

- See [Workflow Stages](#workflow-stages) for more information.

**Type**

- `Feature`: New feature additions.
- `Bug`: Bug within the current codebase.
- `Experimentation`: Running experiments on the current codebase.
- `Research`: Researching a topic - no development involved.

### Creating Merge Requests

Each merge request **must** be tied to an issue from the issue board. To create a new 
MR, follow the instructions below:

1. Create an issue in the issue board and click the issue name to open the issue page.
2. Click the blue `Create merge request` button.
    - (_Optional_) Rename the branch name to something shorter, yet understandable.
3. Click the blue `Create merge request` button.
4. Update the description (if needed) and assign the MR to yourself.
5. Click the blue `Create merge request` button.

Merge requests **must** be created from the parent `dev` branch - **not** `main`.

### Code Reviews

Each MR **needs** to be reviewed by the other members of the project to ensure that the
new code additions are complete and stable. MRs into the `dev` branch require approval
from at least one other member of the team in order to be merged. Reviewers should
look through the code of the MR and leave comments (preferably linked to a line within 
the code). Reviewers should look for areas to reformat, documentation, poor naming
conventions, incorrect code, etc. Some of these can be auto-solved following the 
package usage in [Code Quality](#code-quality).

No branch should be merged into `dev` before going through a review first.

### Merge Process

All feature branches should be merged into `dev` alone. The only branch that should be
merged into `main` is `dev`. This is to ensure the code being pushed to our 
public-facing GitHub is stable, well-documented, and well-tested. The code within `dev`
should also be up to par, but is more lenient as it is internal facing. All merges into
`main` should be marked as a new `version` of SimHarness as only high-quality, large
updates will be getting pushed to `main`.

### Code Quality

Before merging into `dev`, a new MR **must** adhere to the following rules:

- Linting with [`flake8`](https://flake8.pycqa.org/en/latest/).
- Code-formatting with [`black`](https://github.com/psf/black).
- Passing Python [unit tests](#running-unit-tests)
- Documentation creation.

These are all handled by our GitLab CI pipeline automatically using the git `.prehooks`.
The pipeline stages can be tested locally as well to ensure that they are passed on the
remote side (explained in [Using Pre-commit](#using-pre-commit)).


### Using Pre-commit (**Highly Recommended**)

If you'd like, you can install [pre-commit](https://pre-commit.com/) to run linting and 
code-formatting before you are able to commit. This will ensure that you pass this 
portion of the remote pipelines when you push to your merge request.

```shell
pre-commit install
```

Now, every time you try to commit, your code that you have staged will be linted by 
`flake8` and auto-formatted by `black`. If the linting doesn’t pass pre-commit, it will 
tell you, and you’ll have to make those changes before committing those files. If 
`black` autoformats your code during pre-commit, you can view those changes and then 
you’ll have to stage them. Then you can commit and push.

Pre-commit can also be run manually on all files without having to commit.

```shell
pre-commit run --all-files
```

### Running Unit Tests

There are also unit tests that need to be passed, and to make sure you are passing those
 locally (before pushing to your remote branch and running the pipeline) you can run the
  following command in the root directory:

```shell
poetry run pytest
```

This will search for all `test_*.py` files and run the tests held in those files.
