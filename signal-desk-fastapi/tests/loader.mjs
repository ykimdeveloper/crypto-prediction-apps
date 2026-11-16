// resolve extensionless relative imports like Vite does
export async function resolve(specifier, context, next) {
  if ((specifier.startsWith('./') || specifier.startsWith('../')) && !/\.[a-z]+$/.test(specifier)) {
    return next(specifier + '.js', context);
  }
  return next(specifier, context);
}
