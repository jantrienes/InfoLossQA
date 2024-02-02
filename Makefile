CODE=src
TESTS=tests

export OPENAI_API_KEY=''

format:
	black ${CODE} ${TESTS} notebooks
	isort ${CODE} ${TESTS}

test:
	pytest --cov-report html --cov=${CODE} ${CODE} ${TESTS}

lint:
	pylint --recursive=y --disable=R,C ${CODE} ${TESTS}
	black --check ${CODE} ${TESTS}
	isort --check-only ${CODE} ${TESTS}

lintci:
	pylint --recursive=y --disable=W,R,C ${CODE} ${TESTS}
	black --check ${CODE} ${TESTS}
	isort --check-only ${CODE} ${TESTS}
