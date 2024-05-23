def task_clean_untracked():
    """clean untracked files"""
    return {
        "actions": [
            "git clean -fdx",
        ]
    }


def task_doc():
    """gen documentation"""
    return {"actions": ["""make -e -C docs/ SPHINXOPTS="-D language='ru'" html"""]}
