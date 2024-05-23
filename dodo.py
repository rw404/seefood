def task_clean_untracked():
    """clean untracked files"""
    return {
        "actions": [
            "git clean -fdx",
        ]
    }


def task_doc_ru():
    """gen documentation (rus)"""
    return {"actions": ["""make -e -C docs/ SPHINXOPTS="-D language='ru'" html"""]}


def task_doc_eng():
    """gen documentation (eng)"""
    return {"actions": ["""make -e -C docs/ html"""]}
